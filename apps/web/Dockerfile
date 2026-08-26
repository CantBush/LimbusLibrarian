FROM node:22-alpine AS build
WORKDIR /web
COPY package.json ./
RUN npm install
COPY . .
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
