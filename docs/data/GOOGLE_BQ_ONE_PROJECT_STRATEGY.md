# One-Project Google BQ Strategy

## Core idea

For Google behavioral interviews, it is usually better to have one deep, flexible project story than many shallow stories.

A strong primary story is better because:

- It survives follow-up questions.
- It sounds real instead of memorized.
- It keeps my narrative consistent.
- It lets the interviewer see judgment, trade-offs, and learning in depth.

The risk is repeating myself. The solution is not to collect many random stories. The solution is to learn how to rotate the same project through different lenses.

This document turns my AI video commerce recommender project into:

- `1` main story
- `6` Google BQ angles
- likely follow-up questions
- concise answer patterns

---

## The main project story

### Project summary

I built an AI-powered video commerce recommendation system and deliberately pushed it beyond a toy demo. Instead of stopping at a single recommendation endpoint, I defined three concrete request paths, set baseline SLOs, split the runtime into named services, added asynchronous workers, and used load tests plus profiling to evaluate whether the architecture could behave like a production system.

### Why this is a good primary story

This project contains enough depth for Google BQ because it includes:

- ambiguity at the beginning
- architectural decisions
- measurable trade-offs
- performance problems
- failed assumptions
- iteration based on data
- user-impact thinking
- documentation and system clarity for future collaborators

### The simplest version of the story

I started with a vague goal: build a video-commerce recommender. I made it concrete by defining three core API paths, setting targets like `10,000 RPS` for interactions and `1,000-2,000 RPS` for recommendations, and documenting assumptions such as `100,000` active users and `1,000,000` products. I then split the system into `gateway-api`, `recommendation-service`, `interaction-ingest-service`, background workers, and an offline `model-trainer`, moved retraining off the serving path, added `X-Request-ID`, structured logs, health/metrics endpoints, and load tests, and then used benchmark data to correct a bad scaling assumption. The biggest lesson was that strong engineering is not only about building features. It is about making concrete trade-offs under ambiguity and changing direction quickly when evidence says I am wrong.

---

## How to use one project without sounding repetitive

Do not answer every question with the exact same beginning, conflict, and lesson.

Instead, keep the same project but change:

- the problem you emphasize
- the decision you emphasize
- the trade-off you emphasize
- the mistake you emphasize
- the outcome you emphasize
- the lesson you emphasize

Think of the project as a story bank, not a single script.

---

## The six BQ angles

## 1. Ambiguity

### Question this can answer

- Tell me about a time you worked through unclear requirements.
- Tell me about a time you had to create structure from ambiguity.

### Best angle

The project started as a broad idea, not a well-scoped engineering problem. I created clarity by defining scope, non-goals, traffic assumptions, and latency/error targets before locking in architecture.

### Strong answer

One example is my AI video commerce recommender project. The original idea was broad enough that it could easily become a feature demo without real engineering discipline. I did not want that, so before optimizing models or adding features, I wrote down three specific API paths: `POST /api/interactions`, `POST /api/recommendations`, and a mixed `90/10` traffic profile. I also documented assumptions like `100,000` users, `1,000,000` products, `70%` of recommendation requests carrying `content_id`, and warm Redis keys for `95%` of active users. From there I set target SLOs: for interactions, `10,000 RPS`, `P95 < 50ms`, `P99 < 100ms`; for recommendations, `1,000-2,000 RPS`, `P95 < 150ms`, `P99 < 300ms`.

That framing changed the quality of the project. Once I had those constraints, it became obvious that recommendation serving, interaction ingest, and retraining had different workload patterns and should not all live on the same path. I chose to define interfaces and performance targets first instead of starting from model experimentation, because the alternative would have optimized for demo quality rather than system behavior. That led me to split the system into `gateway-api`, `recommendation-service`, `interaction-ingest-service`, `content-worker`, `feature-worker`, and an offline `model-trainer` instead of continuing with a single process.

The main thing I learned is that in ambiguous situations, the first valuable output is not code. It is a decision framework that makes trade-offs explicit.

### Detailed follow-up guide

#### Why did you choose those constraints?

This question is testing whether I created structure thoughtfully or arbitrarily.

Suggested answer:

I chose constraints that would force real engineering decisions instead of letting the project stay vague. The alternative was to start building features first and hope the architecture could be cleaned up later. I did not choose that because once recommendation logic, ingest, and model updates are mixed together, it becomes much harder to separate product iteration from system debt. Once I wrote down target latency and throughput expectations for recommendation serving and interaction ingest, it became obvious that I needed to think about critical-path work very differently from asynchronous work. I was not trying to claim those numbers were perfect. I was using them as a decision tool.

#### What did you leave out, and how did you decide that?

This question is testing prioritization and judgment.

Suggested answer:

I left out systems that were real but not first-order for this project, such as payments, merchant tooling, active-active multi-region deployment, and strict exactly-once guarantees. I could have added those to make the system sound more complete, but I did not think they would improve my understanding of the core recommendation loop. My rule was simple: if a component did not materially affect ingest, retrieve, rank, or serve, it was more likely to dilute the project than strengthen it.

#### How did you know your scope was right?

This question is testing whether I can validate framing decisions.

Suggested answer:

I did not assume the scope was "correct" in some absolute sense. I asked whether it was useful. The sign that it was useful was that it immediately clarified design decisions. Once the scope was explicit, I could define service boundaries, write load tests against specific paths, and reason about degraded behavior. If the scope had still left me unable to make architecture or testing choices, that would have been a sign it was not good enough.

#### If the interviewer pushes with "What if your assumptions were wrong?"

Suggested answer:

That is always possible, which is why I treated the assumptions as working hypotheses, not truths. I wanted them to be concrete enough to guide decisions, but still revisable once I had measurements. That mindset mattered later when load-test data disproved some of my performance assumptions.

### Alternative ambiguity story: evolving the recommendation path

This is a second ambiguity story from the same project, but with a narrower focus on online serving design.

#### Why this also qualifies as ambiguity

The ambiguity was not just "how do I make it faster?" The real ambiguity was:

- what should be cached versus recomputed
- how much personalization should happen on the live path
- where to reduce candidate volume
- how to improve throughput without making results too stale

At the beginning, the recommendation path was much simpler: I effectively had one main cache layer, and if that missed, I fell back to live model-driven work. That was easy to reason about, but it created a fuzzy architecture decision: should I keep making the direct path faster, or should I redesign the serving path itself?

#### Strong answer

Another ambiguity story from the same project is how I redesigned the recommendation path itself. My first version was much simpler: I had essentially one cache layer, and when that missed, I went through the live recommendation path and ranking path directly. That was straightforward, but once I cared more seriously about latency and throughput, it stopped being clear what the right optimization strategy was.

The ambiguous part was not just "make it faster." I had to decide where the serving path should be reusable and where it should stay personalized. I considered a few options. One was to rely more heavily on a single final-response cache. I did not like that because exact final results are more likely to become stale and are less reusable across similar requests. Another option was to keep the architecture simple and just push harder on direct inference with more workers. I had already seen evidence that blind scaling was not solving the real bottleneck. A third option was to simplify ranking aggressively, but I did not want to give up too much result quality before I understood where the cost really was.

So I moved to a more layered recommendation path. I kept the final recommendation cache for exact reuse, but I also added a separate candidate cache keyed on a coarser retrieval context so I could reuse retrieval work without freezing the entire final answer. In the live path, I also constrained candidate volume with `k_per_source = min(k * 10, 500)`, merged candidates from multiple sources, and only then sent the thinner candidate set into ranking. That design let me create different serving paths such as `recommendation_cache`, `candidate_cache_then_rank`, and `live_candidates_then_rank`, instead of treating every miss as "do everything again."

I chose that design because it gave me a better balance than the alternatives. It was more reusable than only caching final responses, but more personalized and fresher than over-caching fully ranked results. It was also more efficient than sending the full live path through expensive ranking work on every miss. The lesson for me was that ambiguity in serving systems is often about boundaries, not algorithms. I had to decide what to reuse, what to recompute, and where freshness still mattered.

#### Detailed follow-up guide

##### Why was the original one-cache design not enough?

This question is testing whether I can explain why an earlier simple design broke down.

Suggested answer:

The original design was fine when the project was smaller and the serving path was less performance-sensitive. But once I started caring more seriously about latency and throughput, it was too binary: either I got an exact cache hit, or I paid for the full live path. That meant I was missing a middle layer where I could reuse expensive retrieval work without locking myself into stale final recommendations.

##### Why not just cache the final recommendations more aggressively?

This question is testing whether I understand the freshness and personalization trade-off.

Suggested answer:

I could have tried to solve the problem by leaning harder on final-response caching, but I did not think that was the right long-term answer. Final ranked recommendations are the least reusable part of the pipeline because they are closest to the user-specific, request-specific output. If I cached them too aggressively, I would improve hit rate at the cost of freshness and personalization. Candidate caching was a better middle ground because it reused retrieval work while still letting me rerank with live user features and context.

##### Why not keep the architecture simple and just optimize inference?

This question is testing whether I know when optimization is not enough.

Suggested answer:

Because by that point I no longer believed the problem was only model speed. The structure of the serving path was also part of the problem. If every miss forced me through the full path again, then even a faster model would not fully solve the inefficiency. I needed a better execution shape, not just a faster implementation of the same shape.

##### Why thin candidates before ranking?

This question is testing whether I made a deliberate cost-quality trade-off.

Suggested answer:

Ranking is the expensive part I want to spend carefully, so I did not want to send every possible candidate through it. I used multiple retrieval sources, merged them, capped the volume, and then ranked the thinner set. I chose that because it preserved a richer candidate pool than a single-source shortcut, but avoided the cost of reranking an unnecessarily large set.

##### What is the main lesson from this story?

Suggested answer:

The main lesson is that ambiguity in system design often looks like a performance problem at first, but the real issue is usually architectural. I had to decide which parts of the recommendation path should optimize for reuse and which parts should optimize for freshness and personalization.

---

## 2. Challenging the status quo

### Question this can answer

- Tell me about a time you challenged an existing approach.
- Tell me about a time you did not accept the default path.

### Best angle

The easy version of the project would have been a monolith that handled serving, ingest, and retraining together. I pushed against that and split responsibilities earlier.

### Strong answer

In this project, the default path would have been to keep everything inside one application process because that is the fastest way to get something working. I chose not to do that because I had already defined very different workloads for serving and ingest. For example, the interaction path was aiming for `10,000 RPS` with `P95 < 50ms`, while the recommendation path was lower-throughput but more latency-sensitive and model-heavy. Treating those as one runtime would have hidden the real bottlenecks.

I split the system into `gateway-api`, `recommendation-service`, `interaction-ingest-service`, `content-worker`, `feature-worker`, and `model-trainer`. The gateway validates requests, supports optional API-key auth, applies in-memory per-client rate limiting, and routes traffic. The interaction service returns `202` quickly and pushes events to Kafka or falls back to a Redis stream. The recommendation service handles serving-time retrieval, caching, and ranking, and the trainer keeps retraining out of the live request path.

The result was more complexity upfront, but it made the system much more honest. I could have kept one service and tried to optimize it later, but that would have mixed too many responsibilities into the same latency budget. By splitting early, I could reason separately about user-facing latency, asynchronous throughput, and model lifecycle. It also meant failure modes were clearer: for example, Kafka issues degraded the system instead of taking down all interaction handling.

### Detailed follow-up guide

#### Was that premature optimization?

This question is testing whether I know the difference between architectural foresight and unnecessary complexity.

Suggested answer:

I do not think it was premature optimization because I was not trying to tune obscure bottlenecks before they existed. The alternative would have been one runtime for everything, but the request types were already fundamentally different. Recommendation serving is latency-sensitive. Interaction ingest is write-heavy and tolerant of asynchronous processing. Retraining is expensive and should not interfere with serving. That is not speculative complexity to me. That is drawing boundaries around work with different operational requirements.

#### What downside did you accept when you split the system?

This question is testing whether I acknowledge cost honestly.

Suggested answer:

The obvious downside was more moving parts. Once I split the system, I had more services to reason about, more health surfaces, and more deployment complexity. I accepted that because I thought the long-term clarity was worth it. I would rather manage a bit more explicit complexity than hide important coupling inside one process and discover the cost later under load or failure.

#### Why not keep it simple first?

This question is usually probing whether I have a bias toward over-engineering.

Suggested answer:

I think "keep it simple" is good advice when it preserves truth. In this case, I felt a single-process architecture would have been simple in implementation but misleading in behavior. I still kept many things simple: I did not build every distributed-systems feature, and I kept local deployment approachable. The specific choice I made was to keep deployment simple but execution paths honest. I did not want simplicity to come from pretending very different workloads belonged on the same critical path.

#### If the interviewer asks "Would you make the same choice again?"

Suggested answer:

Yes, although I might stage the split differently depending on time pressure. The key idea I would keep is separating serving-time work from asynchronous and training work as early as possible.

---

## 3. Mistake or failed assumption

### Question this can answer

- Tell me about a mistake you made.
- Tell me about a time you were wrong.
- Tell me about a time data changed your mind.

### Best angle

I assumed adding more workers would improve recommendation throughput. The load tests showed that this assumption was incomplete or wrong.

### Strong answer

One mistake I made was assuming that recommendation performance would improve mainly by adding serving workers. That was a natural instinct, but the benchmark results showed that scaling worker count did not fix the real problem and in some cases made behavior worse. In the load-test artifacts, `recommendations-direct.json` was around `5.4s` average latency with about `15%` failed requests, and `recommendations-direct-w4.json` was even worse at roughly `7.8s` average latency with about `12%` failed requests.

Instead of defending that assumption, I instrumented the request path more carefully. I chose profiling before more scaling changes because adding workers again would only have repeated the same mistake with less clarity. I added stage-level profiling to the recommendation handler, separated timings like cache lookup, candidate generation, metadata lookup, ranking, and total request time, introduced thread caps, and added `RankingBatcher` so short-lived ranking requests could be micro-batched into a single model forward pass.

That changed the system in a meaningful way. In `recommendations-direct-w1-threadcap-batching.json`, the tuned configuration was around `2.1s` average latency with zero failed requests. It was still not at my target, so I would not oversell it, but it proved that the real issue was not simply "too few workers." The deeper lesson was that "more parallelism" is not a serious performance strategy unless I understand where the contention actually is.

### Detailed follow-up guide

#### Why do you think your first assumption was wrong?

This question is testing whether I can diagnose, not just narrate.

Suggested answer:

My first assumption was incomplete because I treated the problem like a generic concurrency issue. The data suggested something more specific: adding workers did not meaningfully relieve the expensive part of the recommendation path, which seemed closer to inference overhead and contention in the ranking stage. That changed my mental model from "I need more workers" to "I need to understand where each millisecond is going."

#### What did you change first, and why?

This question is testing prioritization.

Suggested answer:

I changed observability first. I could have immediately reduced candidate counts or changed model settings, but those would have been guesses. I did not want to optimize based on instinct because that is how performance work becomes noisy and misleading. I added stage-level timing so I could separate cache lookup, candidate generation, metadata lookup, ranking, and total request time. Once I had that, I could make more targeted changes like thread tuning and micro-batching.

#### Why micro-batching?

This question is testing whether my optimization matched the real bottleneck.

Suggested answer:

Micro-batching made sense because the ranking layer was doing repeated small inference work per request. I considered more aggressive alternatives like simplifying the ranking model or cutting candidate quality earlier, but I chose micro-batching first because it attacked overhead without forcing an immediate product-quality trade-off. If I could combine short-lived ranking requests into a single forward pass, I could reduce overhead and use the model path more efficiently while preserving the existing ranking logic.

#### Was the problem fully solved?

This question is testing realism and humility.

Suggested answer:

No, and I would be careful to say that directly in an interview. The system improved in a meaningful way, but it was still above my target latency. I see that as a good example of honest engineering: I improved the system, I understood it better, and I also learned where the remaining work was. My next steps would be earlier candidate pruning, stronger cache strategy, and reducing heavy operations on the request path.

#### If the interviewer asks "What did you learn personally?"

Suggested answer:

I learned that performance problems punish vague thinking. "More parallelism" sounded plausible, but it was not a good enough explanation. I now trust instrumentation much earlier in the process.

---

## 4. User focus

### Question this can answer

- Tell me about a time you put the user first.
- Tell me about a trade-off you made for user experience.

### Best angle

I designed for graceful degradation and fallback behavior instead of requiring every subsystem to be perfect.

### Strong answer

In recommendation systems, it is easy to focus too much on model sophistication and forget that the user experiences the system mainly through speed, relevance, and reliability. I tried to keep that perspective throughout this project.

One concrete choice was designing the system so that degraded quality is preferable to total failure. On the interaction side, the ingest service accepts events asynchronously and returns `202`; if Kafka is available it queues to Kafka, and if not it falls back to a Redis stream instead of dropping the event path completely. On the recommendation side, if collaborative signals are weak or candidate generation is thin, the engine can fall back to content-based, trending, popularity, and small random exploration paths instead of returning nothing. I also added health checks and Prometheus metrics so degraded states were visible rather than silent.

The reason I chose this instead of hard-failing the request was that this is a consumer recommendation surface, not a safety-critical system. In that context, a weaker but still relevant recommendation is usually better than an empty page or a timeout. The principle behind that decision is simple: users do not care whether the architecture is elegant if the product fails. They care whether it remains useful.

### Detailed follow-up guide

#### How do you think about latency versus quality?

This question is testing product judgment.

Suggested answer:

I think of latency and quality as part of the same user experience, not as separate axes. A theoretically better recommendation that arrives too late can still be a worse product outcome. On the user path, I usually prefer a bounded drop in recommendation quality over a timeout or failure. I chose that trade-off here because the alternatives were either expensive synchronous dependencies on the request path or hard failures when signals were weak. That does not mean quality does not matter. It means the system should degrade gracefully rather than collapse.

#### Are low-quality fallbacks risky?

This question is testing whether I understand the product downside of resilience choices.

Suggested answer:

Yes, they are risky if fallback behavior becomes the norm rather than the exception. That is why I would track fallback rate and downstream engagement, not just availability. My goal is not to normalize lower quality. My goal is to protect the user during weak-signal or degraded scenarios while continuing to improve the primary path.

#### How would you measure whether users were actually better off?

This question is testing whether I can connect engineering to outcomes.

Suggested answer:

I would look at both system and product metrics. On the system side, latency percentiles, error rate, cache hit rate, and fallback rate matter. On the product side, I would look at CTR, add-to-cart rate, conversion, session depth, and possibly diversity or repeat-engagement metrics. The key is not to celebrate resilience in isolation. I would want to know whether the user experience remained useful.

#### If the interviewer asks "When would you choose failure over degraded quality?"

Suggested answer:

I would choose failure if degraded output could be actively harmful or misleading. In recommendation systems that bar is usually lower than in safety-critical domains, so bounded degradation is often acceptable. But I still think there should be guardrails around obviously bad or irrelevant fallbacks.

---

## 5. Ownership beyond scope

### Question this can answer

- Tell me about a time you went beyond your job scope.
- Tell me about a time you showed ownership without being asked.

### Best angle

I did not stop at model logic. I also added docs, load testing, metrics, structured logs, health checks, and clearer operational boundaries.

### Strong answer

Even though this was a self-driven project, I did not want it to function like a private experiment that only I could understand. So I went beyond just implementing recommendation logic and invested in the supporting engineering system around it.

I added phase documents, architecture documentation, explicit load-test baselines, `X-Request-ID`, structured JSON logging, Prometheus `/metrics`, `/metrics/system`, `/health`, and a gateway that centralizes auth, rate limiting, and routing. None of that was strictly required to make the recommendation endpoint return data. I chose to spend time on those pieces instead of only adding more recommendation features because I had already seen that unclear system behavior was slowing down technical decisions. Those additions materially improved the project by making it easier to debug, easier to reason about, and easier for another engineer to extend later.

What I learned is that ownership is not only about shipping the obvious feature. It is also about reducing future confusion and maintenance cost.

### Detailed follow-up guide

#### Why did you spend time on docs, metrics, and health checks instead of just shipping features?

This question is testing whether I understand leverage.

Suggested answer:

Because those investments changed the quality of every future engineering decision. I could have spent that time on model features, but at that stage the higher-leverage problem was that I still needed clearer system behavior. Without docs and clear boundaries, the architecture would have been harder to reason about. Without observability, performance work would have been guesswork. Without health checks, degraded behavior would be harder to identify. So even though those things were not directly user-facing features, they increased my ability to build and debug the user-facing system responsibly.

#### Which non-feature investment mattered most?

This question is testing prioritization within ownership.

Suggested answer:

The highest-leverage investment was observability. Once I had stage-level profiling, metrics, and clearer health surfaces, I could stop arguing with my intuition and start making evidence-based changes. Documentation and service boundaries were also important, but observability had the biggest effect on day-to-day engineering quality.

#### How do you avoid overbuilding support systems?

This question is testing whether I can control engineering ambition.

Suggested answer:

I try to ask a simple question: what future confusion or risk is this investment removing? If I cannot answer that concretely, I probably do not need it yet. I do not want supporting infrastructure for its own sake. I want enough structure that the main system becomes easier to evolve, debug, and hand off.

#### If the interviewer asks "How is this leadership if you were working alone?"

Suggested answer:

I think leadership in engineering is often about creating clarity and reducing future friction, not just directing people. In this case I was leading the quality bar of the system, even if I was not managing a team.

---

## 6. Team orientation from a self-driven project

### Question this can answer

- Tell me about a time you supported the team.
- What kind of teammate are you?
- How do you collaborate when you have strong opinions?

### Best angle

I should not pretend this was a large team project. Instead, I should show team orientation through legibility, documentation, transferability, and willingness to make decisions reviewable.

### Strong answer

This project was largely self-driven, so I would not claim a team dynamic that did not exist. But I still think it shows how I operate in a team environment because I deliberately built it in a way that a future collaborator could understand and challenge.

I wrote phase documents with concrete traffic assumptions and exit criteria, documented explicit service boundaries, and made operational assumptions visible. For example, I wrote down that recommendation requests assumed `100,000` users, `1,000,000` products, and a `70%` `content_id` hit ratio, and I documented exactly which endpoints were in scope for baseline testing. I also added observability so future debugging would not depend on hidden context in my head. To me, that is a very practical form of team orientation: making decisions reviewable and ownership transferable.

If I were in a larger team, I would apply the same instinct earlier through design reviews and earlier challenge from peers. I care less about being the person with the most code and more about making the system and decisions understandable, because that scales better than heroics.

### Detailed follow-up guide

#### How do you prove teamwork in a mostly solo project?

This question is testing honesty and transferability.

Suggested answer:

I would not try to fake team dynamics that were not there. Instead, I would show team-oriented behavior through how I made the project reviewable and transferable. I documented decisions, made assumptions explicit, added observability, and created system boundaries that another engineer could understand quickly. I think those are credible signals of how I would operate in a team because they reduce hidden context and make collaboration easier.

#### What would you do differently on a real team?

This question is testing whether I understand the difference between solo speed and team effectiveness.

Suggested answer:

On a real team, I would push some conversations earlier. For example, I would want design review on service boundaries and performance assumptions before I got too deep into implementation. In a self-driven project, I can move faster alone, but in a team environment the better move is often to surface assumptions early so people can challenge them before they become expensive.

#### How do you handle disagreement when you have a strong technical opinion?

This question is testing humility and collaboration style.

Suggested answer:

I try to be opinionated about the problem, not attached to my first solution. The way I handle disagreement is to make the trade-offs explicit, connect them to user impact, and ask what evidence would change our minds. I want discussions to move away from taste and toward falsifiable reasoning. I think that is especially important in systems work, where reasonable people can disagree until metrics or failure analysis clarifies the issue.

#### If the interviewer asks "What kind of teammate are you?"

Suggested answer:

I am the kind of teammate who tries to reduce ambiguity early, make trade-offs visible, and change direction quickly when evidence says I should. I care about technical quality, but I also care about making systems understandable for the next person.

---

## The one-project answer map

If I get a Google BQ question, I should first decide which lens fits best.

### Quick mapping

- unclear requirements -> ambiguity
- disagreeing with default approach -> challenging the status quo
- failure / weakness / mistake -> failed assumption
- user-first / product judgment -> user focus
- leadership / ownership -> beyond scope
- collaboration / Googleyness / teamwork -> team orientation

---

## What to memorize and what not to memorize

### Memorize

- the project setup in 2 to 3 sentences
- the 6 lenses
- 3 to 5 concrete decisions
- 2 to 3 measurable outcomes
- 2 mistakes or limitations
- 1 honest reflection for each angle

### Do not memorize

- full paragraphs
- polished slogans
- identical opening sentences
- inflated claims about team dynamics

---

## My best reusable facts from this project

These are good anchors to mention naturally when relevant:

- I defined baseline SLOs and traffic assumptions before finalizing architecture.
- I split the system into gateway, recommendation serving, interaction ingest, workers, and offline training.
- I moved retraining off the live serving path.
- I treated Kafka as important but not as a single point of failure.
- I added request IDs, structured logs, metrics, and health checks.
- I used load testing and profiling to challenge my own scaling assumptions.
- I improved one direct recommendation benchmark path from roughly `5.4s` average latency with failures to roughly `2.1s` average latency with zero failures after tuning and batching.

---

## A simple answer template

When answering, I can use this structure:

1. Start with the version of the project that matches the question.
2. Explain the specific challenge, not the whole project history.
3. Describe the decision or trade-off I made.
4. Give the outcome or evidence.
5. End with what I learned.

### Example skeleton

In my AI video commerce recommender project, the part most relevant here was ____. The challenge was ____. I decided to ____ because ____. The result was ____. What I learned was ____.

---

## What to avoid in Google BQ answers

- Do not sound like I memorized a script.
- Do not tell the full project history every time.
- Do not exaggerate teamwork that did not happen.
- Do not hide mistakes.
- Do not describe only technology without decision-making.
- Do not give outcomes without explaining trade-offs.

---

## Final advice

My best strategy is:

- use this recommender project as my main story
- prepare the 6 angles deeply
- keep 2 or 3 backup stories for areas this project does not cover well

That is stronger than trying to memorize many small stories, because deep stories are much more robust under Google-style follow-up.

---

## Related file

For broader question/answer examples, see:

- `GOOGLE_BQ_PREP.md`
