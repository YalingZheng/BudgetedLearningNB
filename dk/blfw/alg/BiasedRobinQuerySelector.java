package dk.blfw.alg;

//import java.util.Arrays;
import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
//import weka.estimators.Estimator;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;
import dk.blfw.impl.NaiveBayesAttrUpdatable;

import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.FastVector;

public class BiasedRobinQuerySelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static final int seed=1234;
	private Random random= new Random(seed);
	
	private boolean inited=false;
	private int lastai=-1;
	
    private double lOld = -1;
    private double lNew = -1;
    private int whichAttribute = -1;
	
	
	public BiasedRobinQuerySelector() {
		super();
	}
	
	@Override
	public IntfQuery propose(EnumMap<QueryRequest, Object> context, Map optional) {		
		Instances pool= (Instances) context.get(QueryRequest.P);
		NaiveBayesAttrUpdatable c= (NaiveBayesAttrUpdatable) context.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		
		if (!inited){
			inited=true;
			lOld = 1;
			lNew = 1;
			whichAttribute = 0;
		}	
		
		if (lastai>=0){
			//receive reward for last action
			//update

			try {

				calclNew(pool,c);
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
		/**
		 * quit util find a missing attribute
		 */
		while(true){
			int ai= drawAttr(pool.numAttributes());
			int ii= drawInst(pool, c);
			//System.out.println("here4 " + ai + " " + ii);

			if  (pool.instance(ii).isMissing(ai)) {
				lastai=ai;
				lOld = lNew;
				//System.out.println("here3");
				return new PhonyAttrQuery(ii,ai);
			}
		}			


		
	}


	private double calclNew(Instances pool, NaiveBayesAttrUpdatable c) throws Exception {
		
		EvaluationUtils eu= new EvaluationUtils();
		String[]	classnames= new String[pool.classAttribute().numValues()];
		for(int i=0; i<classnames.length; i++){
			classnames[i]= "class"+ Integer.toString(i);
		}
		
		FastVector preds=eu.getTestPredictions(c, pool);
		
		ConfusionMatrix cm= new ConfusionMatrix(classnames);
		
		cm.addPredictions(preds);
		
		double x_i= cm.errorRate();
		return x_i;
	}
	
	private int drawAttr(int numAttributes){
		if(lNew >= lOld)
		{
			whichAttribute = (whichAttribute + 1) % numAttributes;
		}
		
		return whichAttribute;
	}
	
	private int drawInst(Instances p, NaiveBayesAttrUpdatable c){
		double rand = dk.blfw.Global.random.nextDouble();
		int myLabel = 0;
		int i = 0;
		double soFar = 0;
		boolean found = false;
		while(!found)
		{
			soFar += c.getClassDistribution().getProbability(i);
			if(rand <= soFar)
			{
				myLabel = i;
				found = true;
			}
			i++;
		}
			
		int j = -1;
		while(true)
		{
			j = dk.blfw.Global.random.nextInt(p.numInstances());
			if  (p.instance(j).classValue() == myLabel) {
				return j;
			}	
		}
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
}
