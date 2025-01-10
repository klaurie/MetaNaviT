FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.* ./
RUN npm install

# Build the application
COPY . .
RUN npm run build

# ====================================
FROM build AS release

CMD ["npm", "run", "start"]