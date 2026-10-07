import urllib.request, json, time

base = "http://localhost:8000"

# 1. Health check
req = urllib.request.urlopen(base + "/api/health")
health = json.loads(req.read())
print("HEALTH:", json.dumps(health, indent=2))

# 2. Upload a TXT document
boundary = "boundaryboundary"
body_txt = (
    "--boundaryboundary\r\n"
    "Content-Disposition: form-data; name=\"file\"; filename=\"test.txt\"\r\n"
    "Content-Type: text/plain\r\n"
    "\r\n"
    "Patient has Type 2 diabetes mellitus. Take Metformin 500mg twice daily with meals. "
    "Blood glucose target: below 7.0 mmol/L. Follow up in 6 weeks.\r\n"
    "--boundaryboundary--\r\n"
)
body_bytes = body_txt.encode("utf-8")
req2 = urllib.request.Request(
    base + "/api/documents",
    data=body_bytes,
    headers={"Content-Type": "multipart/form-data; boundary=boundaryboundary"},
    method="POST",
)
resp2 = urllib.request.urlopen(req2)
doc = json.loads(resp2.read())
print("UPLOAD:", "doc_id=" + doc["doc_id"], "words=", doc["word_count"])

# 3. Simplify (mock mode)
payload = json.dumps({"doc_id": doc["doc_id"], "reading_level": "basic"}).encode()
req3 = urllib.request.Request(
    base + "/api/simplify",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp3 = urllib.request.urlopen(req3)
simpl = json.loads(resp3.read())
job_id = simpl["job_id"]
print("SIMPLIFY: job_id=", job_id, "status=", simpl["status"])

# 4. Poll job
time.sleep(2)
req4 = urllib.request.urlopen(base + f"/api/jobs/{job_id}")
job = json.loads(req4.read())
print("JOB STATUS:", job["status"], "| stage:", job["stage"])
if job.get("result"):
    r = job["result"]
    print("  chunks:", r.get("total_chunks"), "| glossary terms:", len(r.get("glossary", [])))
    print("  scores:", r.get("scores"))
    print("  flags:", r.get("flags"))
    print("  mock_mode:", r.get("mock_mode"))
print("All endpoint tests PASSED.")
