package dk.blfw.alg;

import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;

public class RandomQuerySelector extends QuerySelector{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	Random r= new Random(0);
	
	@Override
	@SuppressWarnings("unchecked")
	public IntfQuery propose(EnumMap<QueryRequest, Object> request, Map optional) {
		Instances pool= (Instances) request.get(QueryRequest.P);
		
		//make sure there should be at least one missing value somewhere.
		boolean hasmissing=false;
		for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			if (i.hasMissingValue()) {hasmissing=true; break;}
		}

		if (hasmissing==false){
			return IntfQuery.NONQUERY;
		}
		
		while(true){
			int ii= dk.blfw.Global.random.nextInt(pool.numInstances());
			int ai= dk.blfw.Global.random.nextInt(pool.numAttributes()-1); //the last attribute is class!
			if (pool.instance(ii).isMissing(ai)) {
				return new PhonyAttrQuery(ii,ai);
			}
			
		}

		
	}
	
	

}
