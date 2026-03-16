import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.BASE_URL || "http://recommendation-service:8001";
const vus = Number(__ENV.VUS || 50);
const duration = __ENV.DURATION || "30s";
const k = Number(__ENV.K || 20);

const payload = JSON.stringify({
  user_id: __ENV.USER_ID || "warm_cache_user",
  content_id: __ENV.CONTENT_ID || "video_demo_1",
  context: {
    device: __ENV.DEVICE || "mobile",
    page: __ENV.PAGE || "feed",
    session_id: __ENV.SESSION_ID || "warm-cache-session",
    session_position: Number(__ENV.SESSION_POSITION || 1),
    time_on_page: Number(__ENV.TIME_ON_PAGE || 30),
  },
  k,
});

export const options = {
  vus,
  duration,
  thresholds: {
    http_req_failed: ["rate<0.01"],
  },
};

export function setup() {
  const response = http.post(`${baseUrl}/api/recommendations`, payload, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  check(response, {
    "warmup status is 200": (r) => r.status === 200,
  });

  return { warmed: response.status === 200 };
}

export default function (data) {
  const response = http.post(`${baseUrl}/api/recommendations`, payload, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  check(response, {
    "recommendations status is 200": (r) => r.status === 200,
    "warmup succeeded": () => data.warmed,
  });

  sleep(0.1);
}
