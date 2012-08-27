package dk.blfw.alg;

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

@SuppressWarnings("serial")
public class Exp3D1QuerySelector extends QuerySelector {

	
	private double gamma=0.1;
	
	
	
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

	
	public Exp3D1QuerySelector() {
		super();
	}
	
	
	/**
	 * 
	 * @param g parameter \gamma for Exp3
	 */
	public Exp3D1QuerySelector(double g){
		setGamma(g);
	}

	
	private Exp3 e=null;
	
	private int[] i2a; //map index returned from expert to attr index.
 	
	private boolean inited=false;
	
	
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
			inited=true;
			int K=pool.numAttributes()-1;
			e= new Exp3(gamma, K);
			i2a= new int[K];
			
			for(int i=0,j=0; i<pool.numAttributes()-1;i++){
				if (pool.classIndex()!=i){
					i2a[i]=j;
				}else{
					i2a[i]=++j;
				}
				j++;
			}
		}	
		
		//step 1:
		if (e.getLastT()>=0){
			//receive reward for last action
			//update
			try {
				double r = calcReward(pool, c);
				e.updateWithReward(r);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		//step 2:
		/**
		 * quit util find a missing attribute
		 */
		while(true){
			int ai= e.drawWeighted();
			int ii= e.drawUniform(pool.numInstances());
			if  (pool.instance(ii).isMissing(ai)) {
				e.setLastT(ai);
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
