import http from "k6/http";
import { check } from "k6";

const baseUrl = __ENV.BASE_URL || "http://localhost:8003";
const rate = Number(__ENV.RATE || __ENV.RPS || 1500);
const duration = __ENV.DURATION || "60s";
const preAllocatedVUs = Number(__ENV.PRE_ALLOCATED_VUS || __ENV.VUS || 500);
const maxVUs = Number(__ENV.MAX_VUS || Math.max(preAllocatedVUs, 2000));
const internalKey = __ENV.SECURITY_INTERNAL_SERVICE_KEY || "";

export const options = {
  scenarios: {
    ranking: {
      executor: "constant-arrival-rate",
      rate,
      timeUnit: "1s",
      duration,
      preAllocatedVUs,
      maxVUs,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.005"],
    http_req_duration: ["p(95)<600"],
  },
};

function candidate(index) {
  return {
    product_id: `prod_${String(index).padStart(6, "0")}`,
    collaborative_score: 0.5,
    content_similarity_score: 0.4,
    popularity_score: 0.3,
    combined_score: 0.45,
    source: "loadtest",
  };
}

function buildPayload() {
  const candidates = [];
  for (let i = 0; i < 20; i += 1) {
    candidates.push(candidate(i));
  }
  return JSON.stringify({
    request_id: `ranking-${__ITER}`,
    candidates,
    user_features: {
      user_id: `user_${String(__ITER % 100000).padStart(6, "0")}`,
      total_interactions: 0,
      avg_session_length: 0,
      preferred_categories: [],
      price_sensitivity: 0.5,
      click_through_rate: 0,
      conversion_rate: 0,
      last_active: Date.now() / 1000,
      demographics: {},
    },
    context: {
      device: __ITER % 2 === 0 ? "mobile" : "desktop",
      session_position: (__ITER % 10) + 1,
      time_on_page: __ITER % 180,
    },
    product_metadata_map: {},
    k: 20,
  });
}

export default function () {
  const headers = {
    "Content-Type": "application/json",
  };
  if (internalKey) {
    headers["x-internal-service-key"] = internalKey;
  }

  const response = http.post(`${baseUrl}/internal/rank`, buildPayload(), {
    headers,
  });

  check(response, {
    "ranking status is 200": (r) => r.status === 200,
  });
}
