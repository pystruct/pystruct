// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.0.
//
// AD3 2.0 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.0 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.0.  If not, see <http://www.gnu.org/licenses/>.

#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

namespace AD3 {

#define NEARLY_EQ_TOL(a,b,tol) (((a)-(b))*((a)-(b))<=(tol))
#define NEARLY_BINARY(a,tol) (NEARLY_EQ_TOL((a),1.0,(tol)) || NEARLY_EQ_TOL((a),0.0,(tol)))
#define NEARLY_ZERO_TOL(a,tol) (((a)<=(tol)) && ((a)>=(-(tol))))

#define MAX(a,b) (((a)<(b))? (b) : (a))

#define LOG_ZERO -1000
#define LOG_STABLE(a) (a == 0? LOG_ZERO : log(a))

extern int diff_ms(timeval t1, timeval t2);

extern int diff_us(timeval t1, timeval t2);

extern void InsertionSort(pair<double, int> arr[], int length);

extern int project_onto_simplex_cached(double* x,
				       int d,
				       double r, 
				       vector<pair<double,int> >& y);

extern int project_onto_simplex(double* x, int d, double r);

extern int project_onto_cone_cached(double* x, int d,
				    vector<pair<double,int> >& y);

extern void StringSplit(const string &str,
			const string &delim,
			vector<string> *results);

extern void TrimComments(const string &delim, string *line);

extern void TrimLeft(const string &delim, string *line);

extern void TrimRight(const string &delim, string *line);

extern void Trim(const string &delim, string *line);

} // namespace AD3

#endif
