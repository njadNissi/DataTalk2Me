# Run Talk2Me with Docker

Build the image from the project root:

```sh
docker build -t talk2me .
```

Start the app:

```sh
docker run --rm -p 8501:8501 talk2me
```

Open <http://localhost:8501>. The image includes the Python dependencies and
Graphviz required by the app. Firebase credentials are not copied into the
image; without them, the app uses its existing offline-auth behavior.

To use Firebase, create `.streamlit/secrets.toml` locally with a `[firebase]`
service-account section and mount it read-only at runtime:

```sh
docker run --rm -p 8501:8501 \
  --mount type=bind,src="$PWD/.streamlit/secrets.toml",dst=/app/.streamlit/secrets.toml,readonly \
  talk2me
```

Keep the credentials file private. It is excluded from the Docker build context.
