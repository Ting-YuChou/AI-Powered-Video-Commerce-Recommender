import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const baseUrl = __ENV.BASE_URL || "http://localhost:8000";
const vus = Number(__ENV.VUS || 50);
const duration = __ENV.DURATION || "60s";
const userPool = Number(__ENV.USER_POOL || 100000);
const contentPool = Number(__ENV.CONTENT_POOL || 10000);
const contentHitRatio = Number(__ENV.CONTENT_HIT_RATIO || 0.7);

const rankingPreRankMs = new Trend("ranking_pre_rank_ms");
const rankingFeatureExtractionMs = new Trend("ranking_feature_extraction_ms");
const rankingResponseBuildMs = new Trend("ranking_response_build_ms");
const rankingRerankCandidateCount = new Trend("ranking_rerank_candidate_count");
const rankingModelForwardMs = new Trend("ranking_model_forward_ms");

export const options = {
  vus,
  duration,
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<150"],
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
  const response = http.post(`${baseUrl}/api/recommendations`, buildPayload(), {
    headers: {
      "Content-Type": "application/json",
    },
  });

  check(response, {
    "recommendations status is 200": (r) => r.status === 200,
  });

  if (response.status === 200) {
    const body = response.json();
    const rankingProfile = body?.metadata?.profile?.ranking_profile;
    if (rankingProfile) {
      rankingPreRankMs.add(Number(rankingProfile.pre_rank_ms || 0));
      rankingFeatureExtractionMs.add(Number(rankingProfile.feature_extraction_ms || 0));
      rankingResponseBuildMs.add(Number(rankingProfile.response_build_ms || 0));
      rankingRerankCandidateCount.add(Number(rankingProfile.rerank_candidate_count || 0));
      rankingModelForwardMs.add(Number(rankingProfile.model_forward_ms || 0));
    }
  }

  sleep(0.1);
}
