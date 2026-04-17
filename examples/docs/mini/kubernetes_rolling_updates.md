# Kubernetes Rolling Updates

A rolling update is the default deployment strategy for Kubernetes
Deployments. It replaces pods with the new version a few at a time, so the
service remains available during the rollout. The two parameters that govern
the rate of replacement are `maxUnavailable` and `maxSurge` in the
`spec.strategy.rollingUpdate` block.

## How It Works

When you push a new image to a Deployment, the Deployment controller
creates a new ReplicaSet with the updated pod template. It scales the new
ReplicaSet up and the old one down in steps. Each step is bounded by the
two parameters: `maxUnavailable` caps how far the total ready pods can fall
below the desired replica count, and `maxSurge` caps how many extra pods
can be created above that count while old pods are still running.

## Example

Consider a Deployment with `replicas: 10`, `maxUnavailable: 1`, and
`maxSurge: 1`. At any moment during the rollout, between 9 and 11 pods are
running, and at most one is unavailable. The controller creates one new
pod, waits for it to become `Ready`, then terminates one old pod, and
repeats until the rollout is complete. Setting both values to 0 forces a
strict one-by-one replacement with zero surge, which is slower but gives
the calmest behavior under capacity pressure.

## Readiness Probes

Rolling updates are only as safe as the readiness probe. A pod is treated
as healthy as soon as its readiness probe returns success. If the probe is
too lenient — for example, just checking that the port is open — the
controller will happily terminate old pods while the new ones are still
warming caches or opening database connections. A good readiness probe
exercises the real startup path, including any dependency check that would
make a request fail in production.

## Rollback

If a rollout goes bad, `kubectl rollout undo deployment/<name>` restores
the previous ReplicaSet. The Deployment history is bounded by
`spec.revisionHistoryLimit` (default 10). Rolling back is itself a rolling
update: it uses the same `maxUnavailable` and `maxSurge` settings, so a
bad rollout that paused halfway can be reversed without a global outage.

## Pitfalls

Three common mistakes derail rolling updates. First, forgetting to set
resource requests, which causes the scheduler to over-pack nodes and
violate the surge budget. Second, running long-lived connections without
`terminationGracePeriodSeconds` tuned, so clients see resets. Third, tying
the readiness probe to an external dependency that itself is flapping —
the pod will toggle ready/unready in a loop and starve the rollout.
