import http from "k6/http";
import { check, sleep } from "k6";
import exec from "k6/execution";
import { Counter, Rate, Trend } from "k6/metrics";

const baseUrl = __ENV.BASE_URL || "http://localhost";
const mode = __ENV.MODE || "candidate_cache_rank";
const rate = Number(__ENV.RATE || __ENV.RPS || 250);
const duration = __ENV.DURATION || "30s";
const preAllocatedVUs = Number(__ENV.PRE_ALLOCATED_VUS || __ENV.VUS || 600);
const maxVUs = Number(__ENV.MAX_VUS || Math.max(preAllocatedVUs, 1500));
const warmUsers = Number(__ENV.WARM_USERS || maxVUs);
const prewarmVUs = Number(__ENV.PREWARM_VUS || 10);
const runOffset = Number(__ENV.RUN_OFFSET || 0);
const apiKey = __ENV.API_API_KEY || "";

const recommendationErrors = new Rate("recommendation_errors");
const recommendation5xx = new Rate("recommendation_5xx");
const recommendation429 = new Rate("recommendation_429");
const candidateCacheRankResponses = new Rate("candidate_cache_rank_responses");
const recommendationCacheResponses = new Rate("recommendation_cache_responses");
const modelForwardResponses = new Rate("model_forward_responses");
const fallbackResponses = new Rate("recommendation_fallback_responses");
const emptyCandidateResponses = new Rate("empty_candidate_responses");
const successfulLatency = new Trend("recommendation_success_latency", true);
const rankingLatency = new Trend("recommendation_ranking_latency", true);
const modelForwardLatency = new Trend("recommendation_model_forward_latency", true);
const batchRequestCount = new Trend("recommendation_batch_request_count", true);
const batchCandidateCount = new Trend("recommendation_batch_candidate_count", true);
const untrainedFallbackResponses = new Rate("ranking_untrained_fallback_responses");
const responseParseErrors = new Counter("recommendation_response_parse_errors");

const measuredOptions = {
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
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
    recommendation_errors: ["rate<0.005"],
    recommendation_5xx: ["rate<0.001"],
    recommendation_success_latency: ["p(95)<500", "p(99)<750"],
    dropped_iterations: ["count==0"],
    candidate_cache_rank_responses: ["rate>0.99"],
    model_forward_responses: ["rate>0.99"],
    ranking_untrained_fallback_responses: ["rate==0"],
    recommendation_cache_responses: ["rate<0.001"],
  },
};

const prewarmOptions = {
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
  scenarios: {
    prewarm: {
      executor: "shared-iterations",
      vus: Math.min(prewarmVUs, warmUsers),
      iterations: warmUsers,
      maxDuration: __ENV.PREWARM_MAX_DURATION || "10m",
    },
  },
  thresholds: {
    recommendation_errors: ["rate<0.001"],
    empty_candidate_responses: ["rate<0.001"],
  },
};

export const options = mode === "prewarm" ? prewarmOptions : measuredOptions;

function headers() {
  const value = { "Content-Type": "application/json" };
  if (apiKey) value["x-api-key"] = apiKey;
  return value;
}

function userId(index) {
  return `loadtest_user_${String(index).padStart(5, "0")}`;
}

function payloadFor(index, iteration, selectedMode) {
  const live = selectedMode === "live_candidates_rank";
  return JSON.stringify({
    user_id: userId(index),
    content_id: null,
    context: {
      device: "mobile",
      page: live ? `loadtest_live_${iteration}` : "loadtest",
      session_position: selectedMode === "prewarm" ? 0 : 1000000 + runOffset + iteration,
      time_on_page: selectedMode === "prewarm" ? 0 : 2000000 + runOffset + iteration,
      session_id: `loadtest_${index}_${iteration}`,
    },
    k: 20,
  });
}

function observe(response) {
  const isError = response.status !== 200;
  recommendationErrors.add(isError);
  recommendation5xx.add(response.status >= 500);
  recommendation429.add(response.status === 429);
  if (!isError) successfulLatency.add(response.timings.duration);

  let body = null;
  try {
    body = response.json();
  } catch (_) {
    responseParseErrors.add(1);
  }
  const metadata = (body && body.metadata) || {};
  const profile = metadata.profile || {};
  const rankingProfile = profile.ranking_profile || {};
  const modelForwardMs = Number(rankingProfile.model_forward_ms || 0);
  const batchRequests = Number(rankingProfile.batch_request_count || 0);
  const batchCandidates = Number(rankingProfile.batch_candidate_count || 0);
  const candidateCount = Number(profile.candidate_count || metadata.total_candidates || 0);

  candidateCacheRankResponses.add(profile.serving_path === "candidate_cache_then_rank");
  recommendationCacheResponses.add(profile.serving_path === "recommendation_cache");
  modelForwardResponses.add(modelForwardMs > 0);
  untrainedFallbackResponses.add(String(rankingProfile.path || "").includes("fallback_untrained"));
  fallbackResponses.add(Boolean(metadata.fallback || metadata.fallback_reason));
  emptyCandidateResponses.add(candidateCount === 0);
  if (Number(profile.ranking_ms || 0) > 0) rankingLatency.add(Number(profile.ranking_ms));
  if (modelForwardMs > 0) modelForwardLatency.add(modelForwardMs);
  if (batchRequests > 0) batchRequestCount.add(batchRequests);
  if (batchCandidates > 0) batchCandidateCount.add(batchCandidates);

  check(response, {
    "status is 200": (r) => r.status === 200,
    "recommendations are non-empty": () =>
      Boolean(body && Array.isArray(body.recommendations) && body.recommendations.length > 0),
  });
}

export default function () {
  let index;
  let iteration;
  let selectedMode = mode;
  if (mode === "prewarm") {
    index = Number(exec.scenario.iterationInTest) + 1;
    iteration = 0;
  } else {
    index = __VU;
    iteration = __ITER;
    if (mode === "mixed") selectedMode = iteration % 5 === 0 ? "live_candidates_rank" : "candidate_cache_rank";
  }
  const response = http.post(
    `${baseUrl}/api/recommendations`,
    payloadFor(index, iteration, selectedMode),
    { headers: headers(), tags: { workload: selectedMode } },
  );
  observe(response);
  if (mode === "prewarm") sleep(0.01);
}
