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

public class Exp3D2QuerySelector extends QuerySelector {

	
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

	
	public Exp3D2QuerySelector() {
		super();
	}
	
	
	/**
	 * 
	 * @param g parameter \gamma for Exp3
	 */
	public Exp3D2QuerySelector(double g){
		setGamma(g);
	}

	
	private Exp3 e1=null;
	private Exp3 e2=null;
	
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
			int K1=pool.numAttributes()-1;
			int K2=pool.numInstances();
			e1= new Exp3(gamma, K1);
			e2= new Exp3(gamma,K2);
			i2a= new int[K1];
			
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
		if (e1.getLastT()>=0 && e2.getLastT()>=0){
			//receive reward for last action
			//update
			try {
				double r = calcReward(pool, c);
				e1.updateWithReward(r);
				e2.updateWithReward(r);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		//step 2:
		/**
		 * quit util find a missing attribute
		 */
		while(true){
			int ai= e1.drawWeighted();
			int ii= e2.drawWeighted();
			if  (pool.instance(ii).isMissing(ai)) {
				e1.setLastT(ai);
				e2.setLastT(ii);
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

