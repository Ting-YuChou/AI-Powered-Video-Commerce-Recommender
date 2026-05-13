import http from "k6/http";
import { check } from "k6";

const baseUrl = __ENV.BASE_URL || "http://localhost:8000";
const rate = Number(__ENV.RATE || __ENV.RPS || 1000);
const duration = __ENV.DURATION || "60s";
const preAllocatedVUs = Number(__ENV.PRE_ALLOCATED_VUS || __ENV.VUS || 250);
const maxVUs = Number(__ENV.MAX_VUS || Math.max(preAllocatedVUs, 1000));
const userPool = Number(__ENV.USER_POOL || 100000);
const contentPool = Number(__ENV.CONTENT_POOL || 10000);
const contentHitRatio = Number(__ENV.CONTENT_HIT_RATIO || 0.7);
const apiKey = __ENV.API_API_KEY || "";

export const options = {
  scenarios: {
    recommendations: {
      executor: "constant-arrival-rate",
      rate,
      timeUnit: "1s",
      duration,
      preAllocatedVUs,
      maxVUs,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<1000"],
  },
};

function randomInt(max) {
  return Math.floor(Math.random() * max);
}

function buildPayload() {
  const userId = `user_${String(randomInt(userPool)).padStart(6, "0")}`;
  const includeContent = Math.random() < contentHitRatio;

  return JSON.stringify({
    user_id: userId,
    content_id: includeContent
      ? `content_${String(randomInt(contentPool)).padStart(6, "0")}`
      : null,
    context: {
      device: Math.random() < 0.8 ? "mobile" : "desktop",
      page: Math.random() < 0.6 ? "home" : "detail",
      session_position: randomInt(10) + 1,
      time_on_page: randomInt(180),
    },
    k: 20,
  });
}

export default function () {
  const headers = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  const response = http.post(`${baseUrl}/api/recommendations`, buildPayload(), {
    headers,
  });

  check(response, {
    "recommendations status is 200": (r) => r.status === 200,
  });
}
