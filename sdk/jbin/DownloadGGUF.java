
import java.io.File;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Download a GGUF model from HuggingFace with the same naming conventions as
 * llama-cli.
 */
public class DownloadGGUF {
	final static Path MODELS_BASE = File.separatorChar == '/'
			? Paths.get(System.getProperty("user.home"), ".cache", "llama.cpp")
			: Paths.get(System.getProperty("user.home"), "AppData", "Local", "llama.cpp");

	public static void main(String[] args) throws Exception {
		if (args.length == 0) {
			System.err.println("Download a quantized model (default is Q4_K_M)\n" + //
					"Usage: " + DownloadGGUF.class.getSimpleName() //
					+ " <hf repo> [<quantization>]\n" //
					+ "e.g. unsloth/Qwen3-4B-Instruct-2507-GGUF Q4_K_M");
		}

		String hfRepo = args[0];
		String quantization = "Q4_K_M";
		if (args.length > 1)
			quantization = args[1];

		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
		String localFileName = hfRepo.replace("/", "_") + "_" + fileName;
		Path localFile = MODELS_BASE.resolve(localFileName);
		if (Files.exists(localFile))
			throw new IllegalStateException(localFile + " already exists, delete it first.");
		Files.createDirectories(localFile.getParent());

		URL url = new URI("https://huggingface.co/" + hfRepo + "/resolve/main/" + fileName + "?download=true").toURL();

		try (InputStream in = url.openStream()) {
			System.out.println("Starting download of " + url);
			long begin = System.currentTimeMillis();
			Files.copy(in, localFile);
			System.out.println("Download completed after " + (System.currentTimeMillis() - begin) / 1000 + " s");
		}
	}

}
