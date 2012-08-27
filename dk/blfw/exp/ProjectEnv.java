package dk.blfw.exp;

import java.io.File;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import dk.app.core.IntfEnv;

public class ProjectEnv implements IntfEnv {
	


	public enum Defaults {
		ARGV("ARGV",null), //map to command line arguments
		ENGINE("ENGINE", null), //map to the engine
		EXECKIND("EXECKIND",null), //map to ExecKind
		
		ROOTDIR("ROOT","."), //map to the root dir.
		BATCHDIR("PATH_BATCH", "batch"+ File.separator+"jobs"), //map to the dir where the batch directory should live
		DATADIR("PATH_DATA", "data"),  //map to "data"
		OUTPUTDIR("PATH_OUTPUT","output"),  //map to "output"
		RESULTDIR("PATH_RESULT", "result"),  //map to "result"
		INSTRFILE("NAME_INSTR_FILE","instruction.xml"), //map to "instruction.xml"
		JOBFILE("NAME_JOB_FILE","job.xml"); //map to "job.xml"
	
		private String key;
		private Object value; 
		Defaults(String k, Object v){
			key=k;
			value=v;
		}
		
		public String getKey(){
			return key;
		}
		
		public Object getValue(){
			return value;
		}
		
	}
	
	HashMap ht= new HashMap();
	
	@SuppressWarnings("unchecked")
	private void addDefaultEntry(){
		
		for (Defaults key : EnumSet.allOf(Defaults.class)) {
			ht.put(key, key.getValue());
		}
	}
	
	public ProjectEnv(){
		addDefaultEntry();
	}
	
	
	
	
	public Object get(Object name) {
		return ht.get(name);
	}



	public void clear() {
		ht.clear();
	}

	private ProjectEnv(HashMap ht){
		this.ht=ht;
	}
	
	@Override
	public Object clone() {
		return new ProjectEnv((HashMap) ht.clone());
	}
	

	
	public boolean containsKey(Object key) {
		return ht.containsKey(key);
	}

	public boolean containsValue(Object value) {
		return ht.containsValue(value);
	}



	public Set entrySet() {
		return ht.entrySet();
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof IntfEnv) {
			IntfEnv ie = (IntfEnv) o;
			ie.equals(ht);
		}
		return false;
	}

	@Override
	public int hashCode() {
		return ht.hashCode();
	}

	public boolean isEmpty() {
		return ht.isEmpty();
	}



	public Set keySet() {
		return ht.keySet();
	}

	@SuppressWarnings("unchecked")
	public Object put(Object key, Object value) {
		return ht.put(key, value);
	}

	@SuppressWarnings("unchecked")
	public void putAll(Map t) {
		ht.putAll(t);
	}

	public Object remove(Object key) {
		return ht.remove(key);
	}

	public int size() {
		return ht.size();
	}

	@Override
	public String toString() {
		return ht.toString();
	}


	public Collection values() {
		return ht.values();
	}



	
	

}
