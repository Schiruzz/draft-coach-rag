# draft-coach-rag

A retrieval-augmented draft coach for a mobile game's ranked mode: it answers draft questions
and grades your picks using pro players' guides, citing the video and timestamp.

Status: step 0 of phase 1 (repository and local databases).

## Quick start
1. Open the repository in GitHub Codespaces: the dev container installs the package
   and starts MongoDB and Qdrant with Docker Compose.
2. Check that both databases answer: `python scripts/check_services.py`
3. Run the tests: `pytest`

## License
Code: MIT. Transcripts and data derived from third-party sources keep their original terms.

---
This material is unofficial and is not endorsed by Supercell. For more information see
Supercell's Fan Content Policy: www.supercell.com/fan-content-policy.