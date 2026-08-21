# GitHub Private Repository Upload

Recommended repository name:

```text
stream-batch-lakehouse-ai-portfolio
```

Recommended description:

```text
End-to-end stream-batch big data portfolio with Kafka, Flink, MinIO, Paimon, Spark, reliability playbooks, and real-time AI recommender services.
```

Recommended permissions:

- Visibility: private.
- Default branch: `main`.
- Owner: repository creator.
- Collaborators: prefer `Read` or `Triage`; grant `Write` only when a collaborator needs to push code.
- Branch protection: disable force-push on `main` and require pull requests before merging when collaborators are added.

## Option A: GitHub CLI

```powershell
Set-Location "C:\path\to\stream-batch-lakehouse-ai-portfolio-redacted"
gh auth login
gh repo create yubohann/stream-batch-lakehouse-ai-portfolio --private --description "End-to-end stream-batch big data portfolio with Kafka, Flink, MinIO, Paimon, Spark, reliability playbooks, and real-time AI recommender services." --source . --remote origin --push
```

## Option B: GitHub Token

```powershell
Set-Location "C:\path\to\stream-batch-lakehouse-ai-portfolio-redacted"
$env:GITHUB_TOKEN="<TOKEN_WITH_REPO_SCOPE>"
$owner="yubohann"
$repo="stream-batch-lakehouse-ai-portfolio"
$body = @{
  name = $repo
  description = "End-to-end stream-batch big data portfolio with Kafka, Flink, MinIO, Paimon, Spark, reliability playbooks, and real-time AI recommender services."
  private = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers @{
  Authorization = "Bearer $env:GITHUB_TOKEN"
  Accept = "application/vnd.github+json"
} -Body $body -ContentType "application/json"

git remote add origin "https://github.com/$owner/$repo.git"
git push -u origin main
```

## Existing Repository Push

Use this path when the repository already exists on GitHub:

```powershell
Set-Location "C:\path\to\stream-batch-lakehouse-ai-portfolio-redacted"
git remote add origin "https://github.com/yubohann/stream-batch-lakehouse-ai-portfolio.git"
git branch -M main
git push -u origin main
```
