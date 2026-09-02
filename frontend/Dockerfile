FROM node:26-alpine AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable && corepack prepare pnpm@10.28.2 --activate

# Install dependencies only when needed
FROM base AS deps
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm config set store-dir /pnpm/store \
    && pnpm install --frozen-lockfile

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_DEFAULT_FLOW_MODE=mixed
ARG NEXT_PUBLIC_API_BASE=http://localhost:2024
ARG NEXT_PUBLIC_LANGGRAPH_API_BASE=http://localhost:2024
ARG FASTAPI_BACKEND_URL=http://localhost:2024
ARG LANGGRAPH_BACKEND_URL=http://langgraph:2024

# Next.js collects completely anonymous telemetry data about general usage.
# Learn more here: https://nextjs.org/telemetry
# Uncomment the following line in case you want to disable telemetry during the build.
ENV NEXT_TELEMETRY_DISABLED 1
ENV NEXT_PUBLIC_DEFAULT_FLOW_MODE=${NEXT_PUBLIC_DEFAULT_FLOW_MODE}
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
ENV NEXT_PUBLIC_LANGGRAPH_API_BASE=${NEXT_PUBLIC_LANGGRAPH_API_BASE}
ENV FASTAPI_BACKEND_URL=${FASTAPI_BACKEND_URL}
ENV LANGGRAPH_BACKEND_URL=${LANGGRAPH_BACKEND_URL}

RUN pnpm build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
# Uncomment the following line in case you want to disable telemetry during runtime.
ENV NEXT_TELEMETRY_DISABLED 1
ENV NEXT_PUBLIC_DEFAULT_FLOW_MODE mixed
ENV NEXT_PUBLIC_API_BASE http://localhost:2024
ENV NEXT_PUBLIC_LANGGRAPH_API_BASE http://localhost:2024
ENV FASTAPI_BACKEND_URL http://localhost:2024
ENV LANGGRAPH_BACKEND_URL http://langgraph:2024

# Don't run as root
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
# set hostname to localhost
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
