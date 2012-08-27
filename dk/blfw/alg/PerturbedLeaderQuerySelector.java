package dk.blfw.alg;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;

public class PerturbedLeaderQuerySelector extends QuerySelector {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private  Random random = new Random();

	private boolean inited=false;
	
	private FEL fel=null;
	
	private int last_ai=-1;
	
	private List excluded= new ArrayList();
	
	private int[] i2a;
	
//	private int count=0;
	
	@SuppressWarnings("unchecked")
	@Override
	public IntfQuery propose(EnumMap<QueryRequest, Object> request, Map optional) {
		Instances pool= (Instances) request.get(QueryRequest.P);
		Classifier c= (Classifier) request.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		//step 0:
		
		if (! inited){
			init(pool);
		}
		
		if (last_ai>=0){
			try {
				//double cost = 1-calcReward(pool, c, last_ai);
				double cost = 1-calcReward(pool, c);
				fel.incur(last_ai,cost);
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		
		while(true){
		
			int ai= fel.draw();
			if (hasMissing(pool, i2a[ai])){
				while(true){

					int ii= dk.blfw.Global.random.nextInt(pool.numInstances());
					if  (pool.instance(ii).isMissing(i2a[ai])) {
						last_ai= ai;
						//System.err.println((++count) + "chosing " + i2a[last_ai]);
						return new PhonyAttrQuery(ii,i2a[ai]);
					}
				}			
			}else {
				//need to reinit the FEL?

				//System.err.println("hmm, is this attr really so good???");
				reinit(ai);
				

			}
		
		}
		
		
	}
	
	@SuppressWarnings("unchecked")
	private void init(Instances pool){
		excluded.add(pool.classIndex());
		last_ai=-1;
		i2a= new int[pool.numAttributes()-excluded.size()];
		for(int i=0,j=0; i<i2a.length;j++){
			if (! excluded.contains(j)){
				i2a[i]=j;
				i++;
			}
		}
		fel = new FEL(i2a.length);
		
		inited=true;
	}
	
	@SuppressWarnings("unchecked")
	private void reinit(int excl){
		
		excluded.add(excl);
		
		i2a= new int[i2a.length-1];
		
		for(int i=0,j=0; i<i2a.length;j++){
			if (! excluded.contains(j)){
				i2a[i]=j;
				i++;
			}
		}
		
		if ((fel!=null) && (excl!=-1))
			fel= fel.shrink(excl);
		else
			fel= new FEL(i2a.length);
		
		//inited=true;

	}
	
	private double calcReward(Instances pool, Classifier c, int attr_i) throws Exception {
		LossFunction MyLossFunction = new LossFunction();
		return 1.0 / MyLossFunction.ConditionalEntropy(pool, attr_i);
		
	}

	private double calcReward(Instances pool, Classifier c) throws Exception {
		Evaluation eval= new Evaluation(new Instances(pool,0));
		eval.evaluateModel(c, pool);
		return 1-eval.errorRate();
		
	}
	
	@SuppressWarnings("unchecked")
	private boolean hasMissing(Instances pool){
		boolean hasmissing=false;
		for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			if (i.hasMissingValue()) {hasmissing=true; break;}
		}
		return hasmissing;
	}
	
	@SuppressWarnings("unchecked")
	private boolean hasMissing(Instances pool, int attr){
		boolean hasmissing=false;
		for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			if (i.isMissing(attr)) {hasmissing=true; break;}
		}
		return hasmissing;
	}

}
