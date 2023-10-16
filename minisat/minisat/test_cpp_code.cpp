#include<bits/stdc++.h>
#include "minisat/gym/GymSolver.h"

using namespace std;

int main(){

	GymSolver S = new GymSolver("./data/uf50-218-tvt/train/uf50-01.cnf", false, 1000000);

	printf("%d",S.propagations);

	return 0;

}