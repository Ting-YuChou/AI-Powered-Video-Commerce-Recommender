import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.BASE_URL || "http://recommendation-service:8001";
const vus = Number(__ENV.VUS || 50);
const duration = __ENV.DURATION || "30s";
const userPool = Number(__ENV.USER_POOL || 200);
const contentIds = (__ENV.CONTENT_IDS || "").split(",").filter(Boolean);
const categories = (__ENV.CATEGORIES || "electronics,fashion,home,beauty,sports")
  .split(",")
  .filter(Boolean);
const devices = (__ENV.DEVICES || "mobile,desktop").split(",").filter(Boolean);
const pages = (__ENV.PAGES || "feed,detail").split(",").filter(Boolean);
const k = Number(__ENV.K || 20);
const skipSetupWarmup = __ENV.SKIP_SETUP_WARMUP === "1";

export const options = {
  vus,
  duration,
  thresholds: {
    http_req_failed: ["rate<0.01"],
  },
};

function randomInt(max) {
  return Math.floor(Math.random() * max);
}

function pick(values) {
  return values[randomInt(values.length)];
}

function buildCoarseContext(userIndex) {
  return {
    device: devices[userIndex % devices.length],
    page: pages[userIndex % pages.length],
    category: categories[userIndex % categories.length],
  };
}

function buildPayload() {
  const userIndex = randomInt(userPool);
  const coarse = buildCoarseContext(userIndex);
  return JSON.stringify({
    user_id: `warm_user_${String(userIndex).padStart(4, "0")}`,
    content_id: contentIds.length > 0 ? contentIds[userIndex % contentIds.length] : null,
    context: {
      ...coarse,
      session_id: `sess_${__VU}_${Date.now()}_${randomInt(100000)}`,
      session_position: randomInt(10) + 1,
      time_on_page: randomInt(180),
    },
    k,
  });
}

export function setup() {
  if (skipSetupWarmup) {
    return { warmed: true };
  }

  for (let i = 0; i < userPool; i += 1) {
    const coarse = buildCoarseContext(i);
    const payload = JSON.stringify({
      user_id: `warm_user_${String(i).padStart(4, "0")}`,
      content_id: contentIds.length > 0 ? contentIds[i % contentIds.length] : null,
      context: {
        ...coarse,
        session_id: `warmup_${i}`,
        session_position: 1,
        time_on_page: 30,
      },
      k,
    });
    const response = http.post(`${baseUrl}/api/recommendations`, payload, {
      headers: { "Content-Type": "application/json" },
    });
    check(response, {
      "warm distribution setup status is 200": (r) => r.status === 200,
    });
  }

  return { warmed: true };
}

export default function (data) {
  const response = http.post(`${baseUrl}/api/recommendations`, buildPayload(), {
    headers: {
      "Content-Type": "application/json",
    },
  });

  check(response, {
    "recommendations status is 200": (r) => r.status === 200,
    "warm distribution setup succeeded": () => data.warmed,
  });

  sleep(0.1);
}
