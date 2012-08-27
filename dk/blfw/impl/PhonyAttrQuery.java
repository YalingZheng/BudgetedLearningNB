package dk.blfw.impl;

import dk.blfw.core.IntfQuery;

public class PhonyAttrQuery implements IntfQuery {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private int fInstIndex=-1, fAttrIndex=-1;
	
	
	public PhonyAttrQuery(){}
	
	public PhonyAttrQuery(int ii, int ai){
		setInstIndex(ii);
		setAttrIndex(ai);
	}

	public void setInstIndex(int instIndex) {
		fInstIndex = instIndex;
	}

	public int getInstIndex() {
		return fInstIndex;
	}

	public void setAttrIndex(int attrIndex) {
		fAttrIndex = attrIndex;
	}

	public int getAttrIndex() {
		return fAttrIndex;
	}

	public boolean isNullQuery() {
		return fInstIndex<0 && fAttrIndex<0;
	}
	
	
	
	
}
