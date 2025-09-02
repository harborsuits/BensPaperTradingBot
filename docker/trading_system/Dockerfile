FROM ghcr.io/your-org/trading-bot-base:latest

ENV PYTHONUNBUFFERED=1
ENV COMPONENT=trading_system

# Add any trading system specific dependencies
COPY docker/trading_system/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8000

ENTRYPOINT ["python", "-m", "trading_bot.main"]
CMD ["--component", "trading_system"] 