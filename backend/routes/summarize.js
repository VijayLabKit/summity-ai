import express from "express";
import multer from "multer";
import { extractPdfText } from "../utils/extractPdfText.js";

const router = express.Router();

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_CHUNK_SIZE = 12000;
const MAX_PARALLEL_CHUNKS = 3;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_FILE_SIZE
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype !== "application/pdf") {
      return cb(new Error("Only PDF files are allowed."));
    }

    cb(null, true);
  }
});

const validSummaryLengths = new Set(["short", "medium", "long"]);
const validFormats = new Set(["paragraph", "bullets"]);
const validLanguages = new Set([
  "English",
  "Hindi",
  "Spanish",
  "French",
  "German",
  "Chinese"
]);

function sanitizeOption(value, allowedValues, fallback) {
  return allowedValues.has(value) ? value : fallback;
}

function getLengthInstruction(summaryLength) {
  const lengthMap = {
    short: "Write 2 to 3 concise sentences.",
    medium: "Write 2 to 3 clear paragraphs.",
    long: "Write a detailed but focused summary with important context."
  };

  return lengthMap[summaryLength] || lengthMap.medium;
}

function createPrompt({ summaryLength, format, language }) {
  const lengthInstruction = getLengthInstruction(summaryLength);
  const formatInstruction =
    format === "bullets"
      ? `Write the summary as bullet points.
Each bullet must be on a separate new line.
Each bullet must start with "- ".
Do not write the bullet summary as one paragraph.`
      : "Write the summary as cohesive paragraphs with natural flow.";
  return `
You are a professional academic and business summarization assistant.

Summarize the provided content in ${language}.

Requirements:
- ${lengthInstruction}
- ${formatInstruction}
- Extract 5 to 8 important topical keywords.
- Return only valid JSON.
- Do not include markdown formatting except simple "- " bullet prefixes when bullet format is requested.
- Do not include explanations outside JSON.

JSON format:
{
  "summary": "string",
  "keywords": ["keyword1", "keyword2"]
}
`.trim();
}

function splitTextIntoChunks(text, maxChunkSize = MAX_CHUNK_SIZE) {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  const chunks = [];
  let currentChunk = "";

  for (const paragraph of paragraphs) {
    if ((currentChunk + "\n\n" + paragraph).length <= maxChunkSize) {
      currentChunk = currentChunk
        ? `${currentChunk}\n\n${paragraph}`
        : paragraph;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk);
      }

      if (paragraph.length > maxChunkSize) {
        for (let i = 0; i < paragraph.length; i += maxChunkSize) {
          chunks.push(paragraph.slice(i, i + maxChunkSize));
        }
        currentChunk = "";
      } else {
        currentChunk = paragraph;
      }
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status) {
  return status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

function safeParseGeminiResponse(data) {
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;

  if (!text) {
    throw new Error("Gemini returned an empty response.");
  }

  try {
    const parsed = JSON.parse(text);

    return {
      summary: typeof parsed.summary === "string" ? parsed.summary.trim() : "",
      keywords: Array.isArray(parsed.keywords)
        ? parsed.keywords.filter((keyword) => typeof keyword === "string").slice(0, 8)
        : []
    };
  } catch {
    throw new Error("Gemini returned a response that could not be parsed as JSON.");
  }
}

async function callGemini(prompt, text, attempt = 0) {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey || apiKey === "your_actual_api_key_here") {
    throw new Error("GEMINI_API_KEY is missing or invalid in the backend .env file.");
  }

  const payload = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `${prompt}\n\nContent:\n${text}`
          }
        ]
      }
    ],
    generationConfig: {
      temperature: 0.3,
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties: {
          summary: {
            type: "STRING"
          },
          keywords: {
            type: "ARRAY",
            items: {
              type: "STRING"
            }
          }
        },
        required: ["summary", "keywords"]
      }
    }
  };

  const response = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    let message = `Gemini API error: ${response.status}`;

    try {
      const errorData = await response.json();
      message = errorData?.error?.message || message;
    } catch {
      message = `${message}. Unable to read error details.`;
    }

    if (isRetryableStatus(response.status) && attempt < 3) {
      const delay = 1000 * Math.pow(2, attempt);
      await sleep(delay);
      return callGemini(prompt, text, attempt + 1);
    }

    throw new Error(message);
  }

  const data = await response.json();
  const parsed = safeParseGeminiResponse(data);

  if (!parsed.summary) {
    throw new Error("Gemini returned an empty summary.");
  }

  return parsed;
}

async function summarizeChunksSequentially(chunks) {
  const chunkPrompt = `
You are summarizing one section of a larger document.

Return only valid JSON:
{
  "summary": "brief section summary",
  "keywords": ["keyword1", "keyword2"]
}
`.trim();

  const results = [];

  for (let i = 0; i < chunks.length; i += MAX_PARALLEL_CHUNKS) {
    const batch = chunks.slice(i, i + MAX_PARALLEL_CHUNKS);
    const batchResults = await Promise.all(
      batch.map((chunk) => callGemini(chunkPrompt, chunk))
    );

    results.push(...batchResults);
  }

  return results;
}

async function processTextWithChunking(text, finalPrompt) {
  if (text.length <= MAX_CHUNK_SIZE) {
    const result = await callGemini(finalPrompt, text);

    return {
      summary: result.summary,
      keywords: result.keywords,
      chunked: false,
      chunkCount: 1
    };
  }

  const chunks = splitTextIntoChunks(text);
  const chunkResults = await summarizeChunksSequentially(chunks);
  const combinedChunkSummaries = chunkResults
    .map((result, index) => `Section ${index + 1}:\n${result.summary}`)
    .join("\n\n");

  const finalResult = await callGemini(finalPrompt, combinedChunkSummaries);

  const mergedKeywords = [
    ...new Set([
      ...finalResult.keywords,
      ...chunkResults.flatMap((result) => result.keywords)
    ])
  ].slice(0, 8);

  return {
    summary: finalResult.summary,
    keywords: mergedKeywords,
    chunked: true,
    chunkCount: chunks.length
  };
}

router.post("/", upload.single("pdf"), async (req, res) => {
  try {
    const rawText = typeof req.body.text === "string" ? req.body.text.trim() : "";
    const summaryLength = sanitizeOption(
      req.body.summaryLength,
      validSummaryLengths,
      "medium"
    );
    const format = sanitizeOption(req.body.format, validFormats, "paragraph");
    const language = sanitizeOption(req.body.language, validLanguages, "English");

    let contentToProcess = rawText;
    let source = rawText ? "text" : "unknown";

    if (req.file) {
      const pdfText = await extractPdfText(req.file.buffer);

      if (rawText) {
        contentToProcess = `${rawText}\n\n${pdfText}`;
        source = "text+pdf";
      } else {
        contentToProcess = pdfText;
        source = "pdf";
      }
    }

    if (!contentToProcess || contentToProcess.trim().length < 10) {
      return res.status(400).json({
        error: "Please paste more text or upload a readable PDF."
      });
    }

    const prompt = createPrompt({
      summaryLength,
      format,
      language
    });

    const result = await processTextWithChunking(contentToProcess, prompt);

    return res.status(200).json({
      summary: result.summary,
      keywords: result.keywords,
      meta: {
        source,
        inputLength: contentToProcess.length,
        chunked: result.chunked,
        chunkCount: result.chunkCount,
        model: GEMINI_MODEL
      }
    });
  } catch (error) {
    console.error("Summarization error:", error.message);

    return res.status(500).json({
      error: error.message || "Unable to generate summary."
    });
  }
});

export default router;