import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.BASE_URL || "http://localhost:8000";
const vus = Number(__ENV.VUS || 200);
const duration = __ENV.DURATION || "60s";
const userPool = Number(__ENV.USER_POOL || 100000);
const productPool = Number(__ENV.PRODUCT_POOL || 1000000);

export const options = {
  vus,
  duration,
  thresholds: {
    http_req_failed: ["rate<0.005"],
    http_req_duration: ["p(95)<50"],
  },
};

function randomInt(max) {
  return Math.floor(Math.random() * max);
}

function randomAction() {
  const roll = Math.random();
  if (roll < 0.7) return "view";
  if (roll < 0.9) return "click";
  if (roll < 0.97) return "add_to_cart";
  return "purchase";
}

function buildPayload() {
  return JSON.stringify({
    user_id: `user_${String(randomInt(userPool)).padStart(6, "0")}`,
    product_id: `prod_${String(randomInt(productPool)).padStart(6, "0")}`,
    action: randomAction(),
    context: {
      page: Math.random() < 0.6 ? "video_recommendations" : "product_detail",
      recommendation_position: randomInt(20) + 1,
      session_id: `sess_${randomInt(500000)}`,
    },
  });
}

export default function () {
  const response = http.post(`${baseUrl}/api/interactions`, buildPayload(), {
    headers: {
      "Content-Type": "application/json",
    },
  });

  check(response, {
    "interactions status is 200": (r) => r.status === 200,
  });

  sleep(0.02);
}
