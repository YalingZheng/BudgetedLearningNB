package dk.blfw.impl;

import dk.blfw.core.IntfAnswer;

public class PhonyAttrAnswer implements IntfAnswer {

	private final int ii;
	private final int ai;
	private final double value;
	
	public PhonyAttrAnswer(PhonyAttrAnswer paa){
		ii=paa.ii;
		ai=paa.ai;
		value=paa.value;
	}

	public PhonyAttrAnswer(int ii, int ai, double value){
		this.ii=ii;
		this.ai=ai;
		this.value=value;
	}

	public int getIi() {
		return ii;
	}

	public int getAi() {
		return ai;
	}

	public double getValue() {
		return value;
	}
	
	
	
}
