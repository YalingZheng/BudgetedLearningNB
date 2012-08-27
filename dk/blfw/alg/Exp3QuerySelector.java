package dk.blfw.alg;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;
import java.util.Vector;


import weka.classifiers.Classifier;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;

public class Exp3QuerySelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static final int seed=1234;
	private Random random= new Random(seed);
	private double gamma=0.05;
	
	
	private boolean inited=false;
	private double[] w=null;
	private int K=0;
	private int lastai=-1;
	
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

	
	public Exp3QuerySelector() {
		super();
	}
	
	
	/**
	 * 
	 * @param g parameter \gamma for Exp3
	 */
	public Exp3QuerySelector(double g){
		setGamma(g);
	}


	@Override
	/**
	 * @Pre: the last attribute is the class.
	 */ 
	public IntfQuery propose(EnumMap<QueryRequest, Object> context, Map optional) {		
		Instances pool= (Instances) context.get(QueryRequest.P);
		Classifier c= (Classifier) context.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		
		if (!inited){
			inited=true;
			K=( (Instances) context.get(QueryRequest.P)).numAttributes()-1;
			w= new double[K];
			for (int i=0; i<w.length;i++){
				w[i]=1;
			}
		}	
		
		double[]  p= getP();
		
		if (lastai>=0){
			//receive reward for last action
			//update

			try {

				double x_i = calcReward(pool, c);
				
				double[] x= new double[K];
				x[lastai]= x_i/ p[lastai];
				
				for(int i=0; i<w.length; i++){
					w[i]=w[i]*Math.exp(gamma*x[i]/K);
				}
				p=getP();
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
		/**
		 * quit util find a missing attribute
		 */
		while(true){
			int ai= drawAttr(p);
			int ii= drawInst(pool.numInstances());

			if  (pool.instance(ii).isMissing(ai)) {
				lastai=ai;
				return new PhonyAttrQuery(ii,ai);
			}
		}			


		
	}

	private double calcReward(Instances pool, Classifier c) throws Exception {
		EvaluationUtils eu= new EvaluationUtils();
		String[]	classnames= new String[pool.classAttribute().numValues()];
		for(int i=0; i<classnames.length; i++){
			classnames[i]= "class"+ Integer.toString(i);
		}
		
		FastVector preds=eu.getTestPredictions(c, pool);
		
		ConfusionMatrix cm= new ConfusionMatrix(classnames);
		
		cm.addPredictions(preds);
		
		double x_i= cm.errorRate();
		return 1-x_i;
	}
	
	private double[] getP(){
		double sum_w=0;
		for (int i=0; i<w.length; i++){
			sum_w+=w[i];
		}
		
		double[] p= new double[w.length];
		for (int i=0; i<w.length; i++){
			p[i]=(1-gamma)*w[i]/sum_w + gamma/K;
		}
		return p;
	}
	
	private int drawAttr(double[] P){
		double[] cum_sum= new double[P.length];
		cum_sum[0]=P[0];
		for(int i=1; i<cum_sum.length; i++){
			cum_sum[i]=cum_sum[i-1]+P[i];
		}
		
		double key= dk.blfw.Global.random.nextDouble();
		int ret= Arrays.binarySearch(cum_sum, key);
		if (ret>=0){ return ret;
		}else{
			ret= -(ret+1);
			return ret;
		}
	}
	
	private int drawInst(int num){
		return dk.blfw.Global.random.nextInt(num);
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
