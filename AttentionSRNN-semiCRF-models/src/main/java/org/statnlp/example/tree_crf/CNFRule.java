package org.statnlp.example.tree_crf;

import java.io.Serializable;

public class CNFRule implements Comparable<CNFRule>, Serializable{
	
	private static final long serialVersionUID = 6195230006004424095L;
	public Label leftSide;
	public Label firstRight;
	public Label secondRight;
	
	public CNFRule(Label leftSide, Label firstRight, Label secondRight){
		this.leftSide = leftSide;
		this.firstRight = firstRight;
		this.secondRight = secondRight;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((firstRight == null) ? 0 : firstRight.hashCode());
		result = prime * result + ((leftSide == null) ? 0 : leftSide.hashCode());
		result = prime * result + ((secondRight == null) ? 0 : secondRight.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (!(obj instanceof CNFRule)) {
			return false;
		}
		CNFRule other = (CNFRule) obj;
		if (firstRight == null) {
			if (other.firstRight != null) {
				return false;
			}
		} else if (!firstRight.equals(other.firstRight)) {
			return false;
		}
		if (leftSide == null) {
			if (other.leftSide != null) {
				return false;
			}
		} else if (!leftSide.equals(other.leftSide)) {
			return false;
		}
		if (secondRight == null) {
			if (other.secondRight != null) {
				return false;
			}
		} else if (!secondRight.equals(other.secondRight)) {
			return false;
		}
		return true;
	}
	
	public String toString(){
		return String.format("%s->%s %s", leftSide, firstRight, secondRight);
	}

	public int compareTo(CNFRule o) {
		int result = 1;
		result *= Label.compare(leftSide, o.leftSide);
		if(result != 0) return result;
		result *= Label.compare(firstRight, o.firstRight);
		if(result != 0) return result;
		result *= Label.compare(secondRight, o.secondRight);
		if(result != 0) return result;
		return 0;
	}
}
