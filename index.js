require('dotenv').config();

const {
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  Client,
  EmbedBuilder,
  GatewayIntentBits,
} = require('discord.js');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const ANALYSIS_PROMPT =
  'Analyze this food image. Return a raw JSON object with these exact keys: calories, protein, carbs, fat, food_name. Ensure values are numbers (except food_name). No markdown, no extra text.';
const FALLBACK_MODELS = [
  'gemini-2.5-flash',
  'gemini-2.0-flash'
];
const MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024;
const REQUIRED_KEYS = ['calories', 'protein', 'carbs', 'fat', 'food_name'];
const SUPPORTED_IMAGE_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/heic',
  'image/heif',
]);

const { DISCORD_TOKEN, GEMINI_API_KEY } = process.env;

if (!DISCORD_TOKEN || !GEMINI_API_KEY) {
  console.error('Missing DISCORD_TOKEN or GEMINI_API_KEY in .env');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

client.once('clientReady', () => {
  console.log(`Logged in as ${client.user.tag}`);
});

client.on('messageCreate', async (message) => {
  if (message.author.bot) {
    return;
  }
    const allowedChannelNames = ['health'];
    if (!allowedChannelNames.includes(message.channel.name)) {
        return;
    }

  const imageAttachment = getFirstImageAttachment(message.attachments);
  if (!imageAttachment) {
    return;
  }

  try {
    await message.channel.sendTyping();
    const imageData = await downloadAttachment(imageAttachment);
    const nutrition = await analyzeFoodImage(imageData.buffer, imageData.mimeType);
    
    const shortcutUrl = buildShortcutUrl(nutrition);
    const replyEmbed = formatNutritionSummary(nutrition);
    const components = [buildShortcutButton(shortcutUrl)];

    await message.reply({
      embeds: [replyEmbed],
      components,
      allowedMentions: { repliedUser: true },
    });
  } catch (error) {
    console.error('Failed to process image:', error);

    let replyMsg = 'I could not analyze that image. Make sure it is a valid food photo in JPEG, PNG, WEBP, HEIC, or HEIF format, then try again.';
    
    if (error?.message?.includes('429 Too Many Requests')) {
      replyMsg = 'Dịch vụ hình ảnh hiện đã hết số lượt chạy miễn phí tốc độ cao (Quá tải yêu cầu). Vui lòng đợi khoảng 1 phút rồi up lại nhé!';
    } else if (error?.message?.includes('503 Service Unavailable')) {
      replyMsg = 'Dịch vụ hình ảnh hiện đang nghẽn mạng cục bộ. Vui lòng đợi vài giây và thử lại!';
    }

    await safeReply(
      message,
      replyMsg
    );
  }
});

client.on('error', (error) => {
  console.error('Discord client error:', error);
});

process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
});


function getFirstImageAttachment(attachments) {
  return attachments.find((attachment) => {
    const mimeType = normalizeMimeType(attachment.contentType, attachment.name);
    return Boolean(mimeType);
  });
}

async function downloadAttachment(attachment) {
  const mimeType = normalizeMimeType(attachment.contentType, attachment.name);

  if (!mimeType) {
    throw new Error('Unsupported or missing image content type.');
  }

  if (attachment.size > MAX_INLINE_IMAGE_BYTES) {
    throw new Error('Image is larger than Gemini inline upload limits.');
  }

  const response = await fetch(attachment.url);

  if (!response.ok) {
    throw new Error(`Failed to download image: ${response.status} ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  if (buffer.length === 0) {
    throw new Error('Downloaded image buffer is empty.');
  }

  if (buffer.length > MAX_INLINE_IMAGE_BYTES) {
    throw new Error('Downloaded image is larger than Gemini inline upload limits.');
  }

  return { buffer, mimeType };
}

async function analyzeFoodImage(buffer, mimeType) {
  let lastError;

  for (const modelName of FALLBACK_MODELS) {
    try {
      const model = genAI.getGenerativeModel({ model: modelName });
      const result = await model.generateContent({
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  data: buffer.toString('base64'),
                  mimeType,
                },
              },
              {
                text: ANALYSIS_PROMPT,
              },
            ],
          },
        ],
        generationConfig: {
          responseMimeType: 'application/json',
          temperature: 0.2,
        },
      });

      const responseText = result.response.text();
      return parseNutritionPayload(responseText);
    } catch (error) {
      console.warn(`[Fallback Warning] Model ${modelName} failed:`, error.message);
      lastError = error;
      continue;
    }
  }

  throw new Error(`All fallback models failed. Last error: ${lastError.message}`);
}

function parseNutritionPayload(rawText) {
  const sanitized = stripCodeFences(String(rawText || '').trim());
  const candidates = [sanitized, extractFirstJsonObject(sanitized)].filter(Boolean);
  let lastError = null;

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      return normalizeNutritionObject(parsed);
    } catch (error) {
      lastError = error;
      continue;
    }
  }

  throw new Error(`Unable to parse or validate Gemini response JSON. Last Error: ${lastError ? lastError.message : 'Unknown'}\nRaw: ${rawText}`);
}

function normalizeNutritionObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Gemini response must be a JSON object.');
  }

  const normalized = {};

  for (const key of REQUIRED_KEYS) {
    if (!(key in value)) {
      throw new Error(`Missing required key: ${key}`);
    }

    if (key === 'food_name') {
      const foodName = String(value[key]).trim();
      if (!foodName) {
        throw new Error('food_name must be a non-empty string.');
      }

      normalized[key] = foodName;
      continue;
    }

    let rawVal = value[key];
    if (rawVal == null || String(rawVal).trim().toLowerCase() === 'null' || String(rawVal).trim().toLowerCase() === 'n/a') {
      rawVal = 0;
    }

    let numberValue;
    if (typeof rawVal === 'number') {
      numberValue = rawVal;
    } else {
      const cleanStr = String(rawVal).replace(/,/g, '').trim();
      const parsed = parseFloat(cleanStr);
      numberValue = isNaN(parsed) ? 0 : parsed;
    }

    if (!Number.isFinite(numberValue)) {
      numberValue = 0;
    }

    normalized[key] = roundToTwo(numberValue);
  }

  return normalized;
}

function buildShortcutUrl(nutrition) {
  const input = encodeURIComponent(JSON.stringify(nutrition));
  return `https://namchtq238.github.io/healthscanner/?input=${input}`;
}

function buildShortcutButton(url) {
  return new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setLabel('\u{1F4E5} Log to Apple Health')
      .setStyle(ButtonStyle.Link)
      .setURL(url)
  );
}

function formatNutritionSummary(nutrition) {
  return new EmbedBuilder()
    .setColor('#2ecc71')
    .setTitle(`Nutrition: ${nutrition.food_name}`)
    .addFields(
      { name: 'Calories', value: formatNumber(nutrition.calories), inline: true },
      { name: 'Protein', value: `${formatNumber(nutrition.protein)} g`, inline: true },
      { name: 'Carbs', value: `${formatNumber(nutrition.carbs)} g`, inline: true },
      { name: 'Fat', value: `${formatNumber(nutrition.fat)} g`, inline: true }
    );
}

async function safeReply(message, content) {
  try {
    await message.reply({
      content,
      allowedMentions: { repliedUser: true },
    });
  } catch (replyError) {
    console.error('Failed to send Discord reply:', replyError);
  }
}

function stripCodeFences(text) {
  return text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
}

function extractFirstJsonObject(text) {
  const start = text.indexOf('{');
  if (start === -1) {
    return null;
  }

  let depth = 0;
  let inString = false;
  let isEscaped = false;

  for (let index = start; index < text.length; index += 1) {
    const char = text[index];

    if (inString) {
      if (isEscaped) {
        isEscaped = false;
        continue;
      }

      if (char === '\\') {
        isEscaped = true;
        continue;
      }

      if (char === '"') {
        inString = false;
      }

      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === '{') {
      depth += 1;
      continue;
    }

    if (char === '}') {
      depth -= 1;

      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }

  return null;
}

function normalizeMimeType(contentType, fileName = '') {
  const normalizedType = String(contentType || '')
    .split(';')[0]
    .trim()
    .toLowerCase();

  if (SUPPORTED_IMAGE_TYPES.has(normalizedType)) {
    return normalizedType;
  }

  const extension = fileName.split('.').pop()?.toLowerCase();

  switch (extension) {
    case 'jpg':
    case 'jpeg':
      return 'image/jpeg';
    case 'png':
      return 'image/png';
    case 'webp':
      return 'image/webp';
    case 'heic':
      return 'image/heic';
    case 'heif':
      return 'image/heif';
    default:
      return null;
  }
}

function roundToTwo(value) {
  return Math.round(value * 100) / 100;
}

function formatNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

// Cuối file, thay toàn bộ phần dưới các hàm helper bằng:

// --- RENDER KEEP-ALIVE SYSTEM (EXPRESS) ---
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    res.send('Discord Health Bot is alive and running with Express!');
});

app.listen(PORT, () => {
    console.log(`Keep-alive Express server listening on port ${PORT}`);
});

setInterval(() => {
    const url = process.env.RENDER_EXTERNAL_URL || `http://localhost:${PORT}`;
    fetch(url)
        .then(() => console.log(`[Keep-Alive] Pinged ${url}`))
        .catch(err => console.error(`[Keep-Alive] Ping failed:`, err.message));
}, 10 * 60 * 1000);

// --- DISCORD LOGIN --- chỉ 1 lần duy nhất
console.log("discord_token", DISCORD_TOKEN.slice(0, 5))
console.log("gemini", GEMINI_API_KEY.slice(0, 5))
client.login(DISCORD_TOKEN)
    .then(() => console.log('Discord login successful ✅'))
    .catch(err => console.error('Discord login failed ❌:', err.message));
