package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppVocabulary;

class VocabularyTests extends AbstractLlmTests {
	VocabularyTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		LlamaCppVocabulary vocabulary = getModel().getVocabulary();

		testEncoding(vocabulary);
	}

	void testEncoding(LlamaCppVocabulary vocabulary) {
		int size = 256;

		// in direct, out direct
		assertVocabulary(vocabulary, //
				ByteBuffer.allocateDirect(size), //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in array, out direct
		assertVocabulary(vocabulary, //
				ByteBuffer.allocate(size), //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in string, out direct
		assertVocabulary(vocabulary, //
				null, //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in direct, out array
		assertVocabulary(vocabulary, //
				ByteBuffer.allocateDirect(size), //
				IntBuffer.allocate(size / Integer.BYTES));
		// in array, out array
		assertVocabulary(vocabulary, //
				ByteBuffer.allocate(size), //
				IntBuffer.allocate(size / Integer.BYTES));
		// in string, out array
		assertVocabulary(vocabulary, //
				null, //
				IntBuffer.allocate(size / Integer.BYTES));
	}

	void assertVocabulary(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer out) {
		assert testTokenizeDetokenize(vocabulary, in, out, "Hello World!");
		assert testTokenizeDetokenize(vocabulary, in, out, "Même si je suis Français, je dis bonjour au monde");
		assert testTokenizeDetokenize(vocabulary, in, out, "ἔορθoι χθόνιοι"); // according to olmoe-1b-7b-0924
		assert testTokenizeDetokenize(vocabulary, in, out, "السلام عليكم"); // according to olmoe-1b-7b-0924
		assert testTokenizeDetokenize(vocabulary, in, out, "¡Hola и أَشْكَرُ мир! 👋🏼🌍");
//		logger.log(INFO, "Vocabulary smoke tests variant PASSED");
	}

	boolean testTokenizeDetokenize(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer buf, String msg) {
		if (in != null)
			in.clear();
		buf.clear();

		logger.log(DEBUG, msg);
		if (in == null) {
			IntBuffer tokens = vocabulary.tokenize(msg);
			buf.put(tokens);
		} else {
			in.put(msg.getBytes(UTF_8));
			in.flip();
			vocabulary.tokenize(msg, buf);
		}
		buf.flip();
		logger.log(DEBUG, logIntegers(buf, 32, ", "));
		String str;
		if (in == null) {
			str = vocabulary.deTokenize(buf);
		} else {
			in.clear();
			vocabulary.deTokenize(buf, in);
			in.flip();
			str = UTF_8.decode(in).toString();
		}
		assert str.equals(msg);
		return true;
	}

	/*
	 * STATIC UTILITIES
	 */
	/**
	 * Writes the beginning of an integer buffer as a string. It has no side effect
	 * on the input buffer.
	 */
	static String logIntegers(IntBuffer in, int max, String separator) {
		StringBuilder sb = new StringBuilder();
		integers: for (int i = in.position(); i < in.limit(); i++) {
			if (i != in.position())
				sb.append(separator);
			if (i == max) {
				sb.append("...");
				break integers;
			}
			sb.append(Integer.toString(in.get(i)));
		}
		return sb.toString();
	}

}
