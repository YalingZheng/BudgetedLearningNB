package dk.blfw.impl;



import dk.blfw.core.IntfAnswer;
import dk.blfw.core.IntfOracle;
import dk.blfw.core.IntfQuery;

import weka.core.Instances;

public final class PhonyAttrOracle implements IntfOracle {

	private Instances fDataSet;

	

	public IntfAnswer answerQuery(IntfQuery q) {
		if (q instanceof PhonyAttrQuery) {
			PhonyAttrQuery pq = (PhonyAttrQuery) q;
			int ii=pq.getInstIndex();
			int ai=pq.getAttrIndex();
			return new PhonyAttrAnswer( ii, ai,getDataSet().instance(ii).value(ai));
		}
		return null;
	}

	public Class[] getSupportedQueries() {
		Class[] a= new Class[]{PhonyAttrQuery.class};
		return a;
	}


	public boolean supportQuery(Object q) {
		if (q instanceof PhonyAttrQuery){
			return true;
		}
		return false;
	}
	
	public PhonyAttrOracle(Instances instances){
		setDataSet(instances);
	}

	public void setDataSet(Instances fDataSet) {
		this.fDataSet = fDataSet;
	}

	public Instances getDataSet() {
		return fDataSet;
	}

}
