package tests.mtmd;

import java.nio.file.Path;
import java.util.Objects;

import org.argeo.jjml.llm.LlamaCppModel;

import tests.llm.AbstractLlmTests;

abstract class AbstractMtmdTests extends AbstractLlmTests {
	private final Path mmprojPath;

	AbstractMtmdTests(LlamaCppModel model, Path mmprojPath) {
		super(model);
		Objects.requireNonNull(mmprojPath);
		this.mmprojPath = mmprojPath;
	}

	protected Path getMmprojPath() {
		return mmprojPath;
	}

}
