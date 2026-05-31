package tests.llm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Downloads a GGUF model (by defaults from HuggingFace) with the same naming
 * conventions as llama-cli. This is meant to be used for prototyping, not as a
 * full-fledged models management solution.
 */
class HfModelCache {
	private final Path modelsBase;

	HfModelCache(Path modelsBase) {
		this.modelsBase = modelsBase;
	}

	HfModelCache() {
		this(getDefaultModelsBase());
	}

	private String getLocalFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
//		String localFileName = hfRepo.replace("/", "_") + "_" + fileName;
		String localFileName = fileName;
		return localFileName;
	}

	private Path getLocalHfRepoBaseDir(String hfRepo) {
		return modelsBase.resolve("models--" + hfRepo.replace("/", "--"));
	}

	public Path getLocalFile(String hfRepoArg) throws IOException {
		if (hfRepoArg.contains(":")) {
			return getLocalFile(hfRepoArg.split(":")[0], hfRepoArg.split(":")[1]);
		} else {
			return getLocalFile(hfRepoArg, "Q4_K_M");
		}
	}

	public Path getLocalFile(String hfRepo, String quantization) throws IOException {
		Path baseDir = getLocalHfRepoBaseDir(hfRepo);
		Path refsFile = baseDir.resolve("refs").resolve("main");
		if (!Files.exists(refsFile))
			return null;
		String ref = Files.readString(refsFile).strip();
		Path modelsDir = baseDir.resolve("snapshots").resolve(ref);
		return modelsDir.resolve(getLocalFileName(hfRepo, quantization));
	}

	/*
	 * STATIC
	 */
	/** The default path where GGUF files are downloaded and searched for. */
	private static Path getDefaultModelsBase() {
		Path defaultModelsBase;
		String os = System.getProperty("os.name").toLowerCase();
		if (os.contains("win")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), "AppData", "Local", "llama.cpp");
		} else if (os.contains("mac") || os.contains("darwin")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), "Library", "Caches", "llama.cpp");
		} else { // Linux / Unix
			defaultModelsBase = Paths.get(System.getProperty("user.home"), ".cache", "huggingface", "hub");
		}
		return defaultModelsBase;
	}

}
