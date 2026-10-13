// The app's version has one place of truth: "version" in app/package.json.
//
//   node scripts/version.mjs check             fails unless every other file agrees with it
//   node scripts/version.mjs check --tag v1.2.3  ... and unless the tag is "v" + that version
//   node scripts/version.mjs set 1.2.3         writes a new version everywhere
//
// app/src-tauri/tauri.conf.json reads the version from package.json itself ("version": "../package.json").
// The others are copies that their tools need: app/package-lock.json, app/src-tauri/Cargo.toml and Cargo.lock
// (package "app"), and backend/pyproject.toml.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repo = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (file) => readFileSync(join(repo, file), "utf8");
const write = (file, text) => writeFileSync(join(repo, file), text);

/** Each copy: how to read it and how to replace it. The patterns match exactly one place per file. */
const copies = [
  {
    file: "app/src-tauri/Cargo.toml",
    pattern: /(\[package\]\r?\nname = "app"\r?\nversion = ")([^"]+)(")/,
  },
  {
    file: "app/src-tauri/Cargo.lock",
    pattern: /(\[\[package\]\]\r?\nname = "app"\r?\nversion = ")([^"]+)(")/,
  },
  {
    file: "backend/pyproject.toml",
    pattern: /(\[project\]\r?\nname = "aivoice"\r?\nversion = ")([^"]+)(")/,
  },
];

function lockVersions() {
  const lock = JSON.parse(read("app/package-lock.json"));
  return [lock.version, lock.packages?.[""]?.version];
}

function problems(version, tag) {
  const found = [];
  if (!/^\d+\.\d+\.\d+$/.test(version)) found.push(`app/package.json: "${version}" is not a version like 1.2.3`);
  for (const { file, pattern } of copies) {
    const match = read(file).match(pattern);
    if (!match) found.push(`${file}: no version found`);
    else if (match[2] !== version) found.push(`${file}: ${match[2]}, expected ${version}`);
  }
  for (const locked of lockVersions()) {
    if (locked !== version) found.push(`app/package-lock.json: ${locked}, expected ${version}`);
  }
  const conf = JSON.parse(read("app/src-tauri/tauri.conf.json")).version;
  if (conf !== "../package.json") {
    found.push(`app/src-tauri/tauri.conf.json: "version" must be "../package.json", found "${conf}"`);
  }
  if (tag !== undefined && tag !== `v${version}`) found.push(`the tag is ${tag}, expected v${version}`);
  return found;
}

const [command, ...rest] = process.argv.slice(2);
const current = JSON.parse(read("app/package.json")).version;

if (command === "check") {
  const tagAt = rest.indexOf("--tag");
  const found = problems(current, tagAt >= 0 ? rest[tagAt + 1] : undefined);
  if (found.length > 0) {
    console.error(`Version mismatch (the truth is app/package.json: ${current}):\n  ${found.join("\n  ")}`);
    console.error("Fix it with: node scripts/version.mjs set <version>");
    process.exit(1);
  }
  console.log(`Version ${current} is consistent.`);
} else if (command === "set" && /^\d+\.\d+\.\d+$/.test(rest[0] ?? "")) {
  const next = rest[0];
  write("app/package.json", read("app/package.json").replace(/("version": ")[^"]+(")/, `$1${next}$2`));
  const lock = read("app/package-lock.json");
  let seen = 0; // the first two "version" keys of a lock file are the project's own
  write("app/package-lock.json", lock.replace(/("version": ")[^"]+(")/g, (all, a, b) => (seen++ < 2 ? `${a}${next}${b}` : all)));
  for (const { file, pattern } of copies) write(file, read(file).replace(pattern, `$1${next}$3`));
  const found = problems(next);
  if (found.length > 0) {
    console.error(`Could not set every copy:\n  ${found.join("\n  ")}`);
    process.exit(1);
  }
  console.log(`Version set to ${next}. Commit, then tag v${next} to release.`);
} else {
  console.error("Usage: node scripts/version.mjs check [--tag vX.Y.Z] | set X.Y.Z");
  process.exit(2);
}
