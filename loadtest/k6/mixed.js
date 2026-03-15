import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.BASE_URL || "http://localhost:8000";
const vus = Number(__ENV.VUS || 200);
const duration = __ENV.DURATION || "60s";
const userPool = Number(__ENV.USER_POOL || 100000);
const productPool = Number(__ENV.PRODUCT_POOL || 1000000);
const contentPool = Number(__ENV.CONTENT_POOL || 10000);

export const options = {
  vus,
  duration,
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<100"],
  },
};

function randomInt(max) {
  return Math.floor(Math.random() * max);
}

function recommendationPayload() {
  return JSON.stringify({
    user_id: `user_${String(randomInt(userPool)).padStart(6, "0")}`,
    content_id: `content_${String(randomInt(contentPool)).padStart(6, "0")}`,
    context: {
      device: Math.random() < 0.8 ? "mobile" : "desktop",
      page: "home",
      session_position: randomInt(10) + 1,
      time_on_page: randomInt(180),
    },
    k: 20,
  });
}

function interactionPayload() {
  return JSON.stringify({
    user_id: `user_${String(randomInt(userPool)).padStart(6, "0")}`,
    product_id: `prod_${String(randomInt(productPool)).padStart(6, "0")}`,
    action: Math.random() < 0.8 ? "view" : "click",
    context: {
      page: "mixed_load_test",
      recommendation_position: randomInt(20) + 1,
      session_id: `sess_${randomInt(500000)}`,
    },
  });
}

export default function () {
  const isRecommendation = Math.random() < 0.1;
  const url = isRecommendation
    ? `${baseUrl}/api/recommendations`
    : `${baseUrl}/api/interactions`;
  const payload = isRecommendation ? recommendationPayload() : interactionPayload();

  const response = http.post(url, payload, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  check(response, {
    "mixed request status is 200": (r) => r.status === 200,
  });

  sleep(0.02);
}
