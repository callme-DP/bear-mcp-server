import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const openapiPath = path.join(root, '.well-known', 'openapi.yaml');
const pluginPath = path.join(root, '.well-known', 'ai-plugin.json');

function normalizeBaseUrl(input) {
  const raw = (input || '').trim();
  if (!raw) {
    throw new Error('请提供公网 URL，例如: https://xxxx.ngrok-free.app');
  }
  const url = raw.replace(/\/+$/, '');
  if (!/^https?:\/\//.test(url)) {
    throw new Error(`URL 必须以 http:// 或 https:// 开头: ${url}`);
  }
  return url;
}

function updateOpenapi(baseUrl) {
  const src = fs.readFileSync(openapiPath, 'utf8');
  const updated = src.replace(/(servers:\n\s*- url:\s*).*/m, `$1${baseUrl}`);
  fs.writeFileSync(openapiPath, updated, 'utf8');
}

function updatePlugin(baseUrl) {
  const src = fs.readFileSync(pluginPath, 'utf8');
  const data = JSON.parse(src);

  data.api = data.api || {};
  data.api.type = 'openapi';
  data.api.url = `${baseUrl}/.well-known/openapi.yaml`;
  data.logo_url = `${baseUrl}/.well-known/logo.svg`;
  data.legal_info_url = `${baseUrl}/legal`;

  fs.writeFileSync(pluginPath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function main() {
  const baseUrl = normalizeBaseUrl(process.argv[2] || process.env.PUBLIC_BASE_URL || '');
  updateOpenapi(baseUrl);
  updatePlugin(baseUrl);

  console.log(`[public-url] updated openapi: ${openapiPath}`);
  console.log(`[public-url] updated plugin: ${pluginPath}`);
  console.log(`[public-url] base url: ${baseUrl}`);
}

main();
