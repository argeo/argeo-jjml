package org.argeo.jjml.llm;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.Objects;

/** Performs de/tokenization natively. */
public class LlamaCppVocabulary {
	private final LlamaCppModel model;

	public LlamaCppVocabulary(LlamaCppModel model) {
		this.model = model;
	}

	// Tokenization
	/**
	 * Tokenize a string encoded as a standard UTF-8 byte array. To use when it
	 * makes more sense to convert on the Java side.
	 * 
	 * @see #doDeTokenizeAsUtf8Array(int[], boolean, boolean)
	 */
	private static native int[] doTokenizeUtf8BytesAsArray(long pointer, byte[] str, int offset, int length,
			boolean addSpecial, boolean parseSpecial);

	private static native int[] doTokenizeUtf8AsArray(long pointer, ByteBuffer str, int offset, int length,
			boolean addSpecial, boolean parseSpecial);

	private static native int doTokenizeUtf8(long pointer, ByteBuffer str, int offset, int length, IntBuffer tokens,
			int pos, int size, boolean addSpecial, boolean parseSpecial);

	/** De-tokenize as a string encoded in standard UTF-8. */
	private static native byte[] doDeTokenizeArrayAsUtf8Bytes(long pointer, int[] tokens, int pos, int size,
			boolean removeSpecial, boolean unparseSpecial);

	private static native byte[] doDeTokenizeAsUtf8Bytes(long pointer, IntBuffer tokens, int pos, int size,
			boolean removeSpecial, boolean unparseSpecial);

	private static native int doDeTokenizeAsUtf8(long pointer, IntBuffer tokens, int pos, int size, ByteBuffer str,
			int offset, int length, boolean removeSpecial, boolean unparseSpecial);

	/*
	 * API
	 */

	public void tokenize(CharSequence str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		CharBuffer chars = CharBuffer.wrap(str);
		ByteBuffer utf8 = UTF_8.encode(chars);
		tokenizeUtf8(utf8, tokens, addSpecial, parseSpecial);
	}

	public IntBuffer tokenize(CharSequence str, boolean addSpecial, boolean parseSpecial) {
		CharBuffer chars = CharBuffer.wrap(str);
		ByteBuffer utf8 = UTF_8.encode(chars);
		int[] arr = tokenizeUtf8(utf8, addSpecial, parseSpecial);
		return IntBuffer.wrap(arr);
	}

	public void tokenize(ByteBuffer utf8, IntBuffer tokens, boolean addSpecial, boolean parseSpecial)
			throws IndexOutOfBoundsException {
		tokenizeUtf8(utf8, tokens, addSpecial, parseSpecial);
	}

	public IntBuffer tokenize(ByteBuffer utf8, boolean addSpecial, boolean parseSpecial) {
		int[] arr = tokenizeUtf8(utf8, addSpecial, parseSpecial);
		return IntBuffer.wrap(arr);
	}

	public void deTokenize(IntBuffer in, ByteBuffer utf8, boolean removeSpecial, boolean unparseSpecial)
			throws IndexOutOfBoundsException {
		deTokenizeUtf8(in, utf8, removeSpecial, unparseSpecial);
	}

	public String deTokenize(IntBuffer in, boolean removeSpecial, boolean unparseSpecial) {
		byte[] bytes = deTokenizeUtf8(in, removeSpecial, unparseSpecial);
		return new String(bytes, UTF_8);
	}

	/*
	 * DEFAULTS
	 */
	final public IntBuffer[] tokenizeMultiple(List<? extends CharSequence> prompts) {
		IntBuffer[] tokenLists = new IntBuffer[prompts.size()];
		for (int i = 0; i < prompts.size(); i++) {
			CharSequence prompt = prompts.get(i);
			IntBuffer tokenList = tokenize(prompt);
			tokenLists[i] = tokenList;
		}
		return tokenLists;
	}

	final public IntBuffer tokenize(CharSequence str) {
		return tokenize(str, false, true);
	}

	final public void tokenize(CharSequence str, IntBuffer tokens) throws IndexOutOfBoundsException {
		tokenize(str, tokens, false, true);
	}

	final public String deTokenize(IntBuffer in) {
		return deTokenize(in, true, true);
	}

	final public void deTokenize(IntBuffer in, ByteBuffer out) throws IndexOutOfBoundsException {
		deTokenize(in, out, true, true);
	}

	/*
	 * UTF-8
	 */

	int[] tokenizeUtf8(ByteBuffer in, boolean addSpecial, boolean parseSpecial) {
		checkInput(in);
		// ensure position is 0
		// ByteBuffer in = str.slice().limit(str.limit() - str.position());
		synchronized (in) {
			int[] tokenArr;
			if (in.isDirect()) {
				tokenArr = doTokenizeUtf8AsArray(model.getAsLong(), in, in.position(), in.remaining(), addSpecial,
						parseSpecial);
				in.position(in.limit());
			} else if (in.hasArray() && !in.isReadOnly()) {
				byte[] arr = in.array();
				tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), arr, in.arrayOffset(), in.remaining(),
						addSpecial, parseSpecial);
				in.position(in.limit());
			} else {// copy
				byte[] copy = new byte[in.remaining()];
				in.get(copy, in.position(), copy.length);
				tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), copy, 0, copy.length, addSpecial,
						parseSpecial);
			}
			return tokenArr;
		}
	}

	void tokenizeUtf8(ByteBuffer str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial)
			throws IndexOutOfBoundsException {
		checkInput(str);
		checkOutput(tokens);
		synchronized (tokens) {// we are writing into this buffer and changing its position
			if (str.isDirect() && tokens.isDirect()) {// optimal
				int count = doTokenizeUtf8(model.getAsLong(), str, str.position(), str.remaining(), tokens,
						tokens.position(), tokens.remaining(), addSpecial, parseSpecial);
				if (count < 0)
					throw new IndexOutOfBoundsException(-count);
				str.position(str.limit());
				tokens.position(tokens.position() + count);
			} else {
				int[] tokenArr = tokenizeUtf8(str, addSpecial, parseSpecial);
				if (tokenArr.length > tokens.remaining())
					throw new IndexOutOfBoundsException(tokenArr.length);
				tokens.put(tokenArr);
			}
		}
	}

	byte[] deTokenizeUtf8(IntBuffer in, boolean removeSpecial, boolean unparseSpecial) {
		byte[] outArr;
		if (in.isDirect()) {
			outArr = doDeTokenizeAsUtf8Bytes(model.getAsLong(), in, in.position(), in.remaining(), removeSpecial,
					unparseSpecial);
			in.position(in.limit());
		} else if (in.hasArray() && !in.isReadOnly()) {
			outArr = doDeTokenizeArrayAsUtf8Bytes(model.getAsLong(), in.array(), in.arrayOffset(), in.remaining(),
					removeSpecial, unparseSpecial);
			in.position(in.limit());
		} else {// copy
			int[] copy = new int[in.remaining()];
			in.get(copy, in.position(), copy.length);
			outArr = doDeTokenizeArrayAsUtf8Bytes(model.getAsLong(), copy, 0, copy.length, removeSpecial,
					unparseSpecial);
		}
		return outArr;
	}

	void deTokenizeUtf8(IntBuffer in, ByteBuffer str, boolean removeSpecial, boolean unparseSpecial)
			throws IndexOutOfBoundsException {
		if (in.isDirect() && str.isDirect()) {
			int count = doDeTokenizeAsUtf8(model.getAsLong(), in, in.position(), in.remaining(), str, str.position(),
					str.remaining(), removeSpecial, unparseSpecial);
			if (count < 0)
				throw new IndexOutOfBoundsException(-count);
			str.position(str.position() + count);
			in.position(in.limit());
		} else {
			byte[] bytes = deTokenizeUtf8(in, removeSpecial, unparseSpecial);
			if (bytes.length > (str.limit() - str.position()))
				throw new IndexOutOfBoundsException(bytes.length);
			str.put(bytes);
		}
	}

	/*
	 * UTILITIES
	 */
	private void checkInput(Buffer in) {
		if (in instanceof IntBuffer)
			if (!ByteOrder.nativeOrder().equals(((IntBuffer) in).order()))
				throw new IllegalArgumentException("Int buffer does not use native byte order");
		Objects.requireNonNull(in, "Input buffer cannot be null");
	}

	private void checkOutput(Buffer out) {
		Objects.requireNonNull(out, "Output buffer cannot be null");
		if (out.isReadOnly())
			throw new IllegalArgumentException("Output buffer is read-only");
		if (out instanceof IntBuffer)
			if (!ByteOrder.nativeOrder().equals(((IntBuffer) out).order()))
				throw new IllegalArgumentException("Int buffer does not use native byte order");
	}
}
