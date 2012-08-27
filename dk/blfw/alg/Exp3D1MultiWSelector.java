package dk.blfw.alg;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;

public class Exp3D1MultiWSelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private double gamma=0.05;
	
	
	
	@Override
	public String[] getOptions() {
	    String [] options;
	    options = new String[1];
	    options[0]="-gamma:"+gamma;
	    return options;
	}

	@Override
	@SuppressWarnings("unchecked")
	public Enumeration listOptions() {

	    Vector newVector = new Vector(1);

	    newVector.addElement(new Option(
		      "\tgamma constant\n"
		      + "\t that defines the weight of random expert",
		      "gamma", 1, "-gamma <value>"));
	    return newVector.elements();
	}



	@Override
	public void setOptions(String[] options) throws Exception {
		if (Utils.getOptionPos("gamma", options)>=0){
			 
			String tmp= (Utils.getOption("gamma", options));
			double g= Double.parseDouble(tmp);
			setGamma(g);
		}
	}



	
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}



	public double getGamma() {
		return gamma;
	}	

	
	public Exp3D1MultiWSelector() {
		super();
	}
	
	
	/**
	 * 
	 * @param g parameter \gamma for Exp3
	 */
	public Exp3D1MultiWSelector(double g){
		super();
		setGamma(g);
	}

	//maintain one weight vector for each class
	private Exp3 []e=null;
	
	private int[] i2a; //map index returned from expert to attr index.
 	
	private boolean inited=false;
	
	private int  lastExp=-1;
	
	private double[] classEstimator =null;
	

	
	private ArrayList<ArrayList<Integer>> classIndexes;
	@Override
	/**
	 * @Pre: the last attribute is the class.
	 */ 
	public IntfQuery propose(EnumMap<QueryRequest, Object> context, Map optional) {		
		Instances pool= (Instances) context.get(QueryRequest.P);
		Classifier c= (Classifier) context.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		//step 0:
		if (!inited){

			int K=pool.numAttributes()-1;
			e = new Exp3[pool.numClasses()];
			for (int i=0; i< e.length; i++){
				e[i]= new Exp3(gamma,K);
			}
			
			//e= new Exp3(gamma, K);
			i2a= new int[K];
			
			for(int i=0,j=0; i<pool.numAttributes()-1;i++){
				if (pool.classIndex()!=i){
					i2a[i]=j;
				}else{
					i2a[i]=++j;
				}
				j++;
			}
			
			classEstimator= new double[pool.numClasses()];
			classIndexes = new ArrayList<ArrayList<Integer>>();
			for (int i=0; i<pool.numClasses(); i++){
				classIndexes.add(i, new ArrayList<Integer>());
			}
			
			for(int i=0; i< pool.numInstances(); i++){
				classEstimator[(int)pool.instance(i).classValue()]++;
				ArrayList<Integer> inst_list= classIndexes.get((int)pool.instance(i).classValue());
				inst_list.add(i);
			}
			
			for (int i=0; i<pool.numClasses(); i++){
				classEstimator[i]= ((double) classEstimator[i])/pool.numInstances();
			}

			
			inited=true;
		}	
		
		

		
		//step 1:
		if (lastExp>=0 && e[lastExp].getLastT()>=0){
			//receive reward for last action
			//update
			try {
				double r = calcReward(pool, c);
				e[lastExp].updateWithReward(r);
			} catch (Exception error) {
				error.printStackTrace();
			}
		}
		
		//step 2:
		/**
		 * random draw  a class according to the class label distribution
		 * exp draw a missing attribute for that class
		 * quit util find a missing attribute
		 */
		
		
		while(true){
			int clazz= Exp3.rollDice(classEstimator);
			
			int ai= e[clazz].drawWeighted();
			
			
			
			int tmp= e[clazz].drawUniform(classIndexes.get(clazz).size());
			int ii= classIndexes.get(clazz).get(tmp);
			
			if  (pool.instance(ii).isMissing(ai)) {
				lastExp= clazz;
				e[clazz].setLastT(ai);
				return new PhonyAttrQuery(ii,i2a[ai]);
			}
			
			
		}			
		
	}

	private double calcReward(Instances pool, Classifier c) throws Exception {
		Evaluation eval= new Evaluation(new Instances(pool,0));
		eval.evaluateModel(c, pool);
//		return eval.SFEntropyGain();
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

}
