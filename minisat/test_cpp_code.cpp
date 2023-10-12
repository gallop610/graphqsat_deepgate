#include<bits/stdc++.h>

#include "minisat/utils/System.h"
#include "minisat/utils/ParseUtils.h"
#include "minisat/utils/Options.h"
#include "minisat/core/Dimacs.h"
#include "minisat/simp/SimpSolver.h"
#include "minisat/gym/GymSolver.h"

using namespace Minisat;

int main(){

	GymSolver S = new GymSolver("./data/uf50-218-tvt/train/uf50-01.cnf", false, 1000000);

	printf("%d",S.propagations);

	return 0;

}