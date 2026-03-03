#!/bin/bash
# Manual deploy script — same steps as the post-receive hook
set -e

cd "$(dirname "$0")/.."
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

echo "==> Building Docker images..."
docker compose build

echo "==> Importing images into k3s..."
docker save anki_mcp-anki-bot | sudo k3s ctr images import --all-platforms -
docker save anki_mcp-anki-sync | sudo k3s ctr images import --all-platforms -

echo "==> Applying k8s manifests..."
kubectl apply -f k8s/anki.yaml

echo "==> Restarting deployments..."
kubectl rollout restart deployment/anki-bot deployment/anki-sync -n anki

echo "==> Waiting for rollout..."
kubectl rollout status deployment/anki-bot -n anki --timeout=120s
kubectl rollout status deployment/anki-sync -n anki --timeout=120s

echo "==> Deploy complete!"
kubectl get pods -n anki
