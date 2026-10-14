FROM python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
ARG AGENT_RAM_BYTES=1073741824
ARG AGENT_SCRATCH_BYTES=1073741824
WORKDIR /app
LABEL org.triad-bluff.protocol="1.0" org.triad-bluff.runtime="2" \
      org.triad-bluff.cpu="1" org.triad-bluff.ram-bytes="${AGENT_RAM_BYTES}" \
      org.triad-bluff.scratch-bytes="${AGENT_SCRATCH_BYTES}" \
      org.triad-bluff.gpu="false" org.triad-bluff.gpu-bytes="0"
COPY src/ ./src/
COPY models/ ./models/
COPY submission.json ./
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 HOME=/tmp
USER 10000:10000
ENTRYPOINT ["python", "src/agent.py"]
