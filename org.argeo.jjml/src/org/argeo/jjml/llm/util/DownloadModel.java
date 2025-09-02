package org.argeo.jjml.llm.util;

import static java.lang.System.Logger.Level.DEBUG;
import static java.lang.System.Logger.Level.INFO;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.System.Logger;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.DoubleConsumer;

import org.argeo.jjml.llm.LlamaCppModel;

/**
 * Download a GGUF model from HuggingFace with the same naming conventions as
 * llama-cli.
 */
public class DownloadModel {
	private final static Logger logger = System.getLogger(LlamaCppModel.class.getName());

	private int bufferSize = 1024 * 4096;

	/** Default location for GGUF model files. */
	private final static Path MODELS_BASE = File.separatorChar == '/'
			? Paths.get(System.getProperty("user.home"), ".cache", "llama.cpp")
			: Paths.get(System.getProperty("user.home"), "AppData", "Local", "llama.cpp");

	/** The default path where GGUF files are downloaded and searched for. */
	public static Path getDefaultModelsBase() {
		return MODELS_BASE;
	}

	private final Path modelsBase;

	public DownloadModel(Path modelsBase) {
		this.modelsBase = modelsBase;
	}

	public DownloadModel() {
		this(getDefaultModelsBase());
	}

	public URL getRemoteUrl(String hfRepo, String quantization) {
		URI uri = getRemoteUri(hfRepo, quantization);
		try {
			return uri.toURL();
		} catch (MalformedURLException e) {
			throw new IllegalArgumentException("Malformed download URL: " + uri, e);
		}

	}

	public URI getRemoteUri(String hfRepo, String quantization) {
		return URI.create("https://huggingface.co/" + hfRepo + "/resolve/main/"
				+ getRemoteFileName(hfRepo, quantization) + "?download=true");
	}

	public String getRemoteFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
		return fileName;
	}

	public String getLocalFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
		String localFileName = hfRepo.replace("/", "_") + "_" + fileName;
		return localFileName;
	}

	public Path getLocalFile(String hfRepo, String quantization) {
		return modelsBase.resolve(getLocalFileName(hfRepo, quantization));
	}

	/*
	 * STATIC
	 */

	/** A simple CLI to use from the OS. */
	public static void main(String[] args) throws Exception {
		if (args.length == 0) {
			System.err.println("Download a quantized model (default is Q4_K_M)\n" + //
					"Usage: " + DownloadModel.class.getSimpleName() //
					+ " <hf repo>:[<quantization>]\n" //
					+ "e.g. allenai/OLMo-2-0425-1B-Instruct-GGUF:Q8_0");
		}

		new DownloadModel().getOrDownloadModel(args[0], new SimpleProgressCallback());
	}

	public Path getOrDownloadModel(String hfRepoArg, DoubleConsumer progressCallback) throws IOException {
		if (hfRepoArg.contains(":")) {
			return getOrDownloadModel(hfRepoArg.split(":")[0], hfRepoArg.split(":")[1], progressCallback);
		} else {
			return getOrDownloadModel(hfRepoArg, "Q4_K_M", progressCallback);
		}
	}

	public Path getOrDownloadModel(String hfRepo, String quantization, DoubleConsumer progressCallback)
			throws IOException {
		Path localFile = getLocalFile(hfRepo, quantization);

		String currentEtag = null;
		if (Files.exists(localFile)) {
			try {
				byte[] buf = (byte[]) Files.readAttributes(localFile, "user:etag").getOrDefault("etag", null);
				if (buf != null)
					currentEtag = new String(buf, StandardCharsets.US_ASCII);
			} catch (IOException e) {
				logger.log(DEBUG, "Cannot read attribute from " + localFile, e);
			}
			if (currentEtag == null) {
				throw new IllegalStateException("File " + localFile + " already exist, remove it first");
			}
		}
		Files.createDirectories(localFile.getParent());

		URL url = getRemoteUrl(hfRepo, quantization);
		HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
		if (currentEtag != null)
			urlConnection.setRequestProperty("If-None-Match", currentEtag);
		urlConnection.connect();
		// TODO somehow show download progress
		try (InputStream in = urlConnection.getInputStream()) {
			int responseCode = urlConnection.getResponseCode();
			double contentLength = urlConnection.getContentLengthLong();
			String etag = urlConnection.getHeaderField("etag");
			if (responseCode == HttpURLConnection.HTTP_NOT_MODIFIED || etag.equals(currentEtag)) {
				logger.log(INFO, "Use unmodified " + hfRepo + ":" + quantization + " file with etag=" + currentEtag);
				return localFile;
			}
			logger.log(INFO, "Starting download of " + url + " to " + localFile);
			logger.log(DEBUG, "Effective URL " + urlConnection.getURL());

			long begin = System.currentTimeMillis();
			try (OutputStream out = Files.newOutputStream(localFile, CREATE, TRUNCATE_EXISTING)) {
				byte[] buf = new byte[bufferSize];
				double downloaded = 0;
				int len;
				while ((len = in.read(buf)) != -1) {
					out.write(buf, 0, len);
					downloaded = downloaded + len;
					if (progressCallback != null)
						progressCallback.accept(downloaded / contentLength);
				}
			}
			// Files.copy(in, localFile, StandardCopyOption.REPLACE_EXISTING);
			if (etag != null) {
				Files.setAttribute(localFile, "user:etag", StandardCharsets.US_ASCII.encode(etag));
			}
			logger.log(INFO,
					"Download completed after " + (System.currentTimeMillis() - begin) / 1000 + " s, etag=" + etag);
		}
		return localFile;
	}
}
