package org.argeo.jjml.llm.util;

import java.io.PrintStream;
import java.util.function.DoubleConsumer;

/**
 * Simple CLI implementation of a progress callback of a value between 0.0 and
 * 1.0. Typically used when loading a model.
 */
public class SimpleProgressCallback implements DoubleConsumer {
	private int lastPerctPrinted = -1;

	private final PrintStream out;

	public SimpleProgressCallback(PrintStream out) {
		this.out = out;
	}

	public SimpleProgressCallback() {
		this(System.err);
	}

	protected void printProgressBar(char[] progressBar) {
		if (out != null)
			out.print("\r" + new String(progressBar));
	}

	@Override
	public void accept(double progress) {
		char[] progressBar = new char[10];
		int perct = (int) (progress * 100);

		if (perct > lastPerctPrinted + 10 //
				|| lastPerctPrinted == -1 //
				|| progress == 1.0) {

			for (int i = 0; i < perct / 10; i++)
				progressBar[i] = '#';
			for (int i = perct / 10; i < 10; i++)
				progressBar[i] = '-';
			printProgressBar(progressBar);

			lastPerctPrinted = perct;
			if (progress == 1.0 && out != null)
				out.print("\n");
		}
	}

}