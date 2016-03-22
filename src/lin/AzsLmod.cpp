/* * * * *
 *  AzsLmod.cpp 
 *  Copyright (C) 2013 Rie Johnson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * * * * */

#include "AzsLmod.hpp"

#define AzsLossType_Num 3
static const char *loss_str[AzsLossType_Num] = {
  "Square", "Logistic", "Unknown", 
}; 

/*--------------------------------------------------------*/
void AzsLmod::reset_weights(int dim, int class_num) {
  reset_ws(); 
  m_w.reform(dim, class_num); 
}           

/*--------------------------------------------------------*/
/* static */
void AzsLmod::_get_deriv_classif(AzsLossType loss_type, 
                       const AzDvect *v_pred, /* prediction values */
                       int cls,               /* true class */
                       /*---  output  ---*/
                       AzDvect *v_deriv) /* 1st order derivatives */
{
  int class_num = v_pred->rowNum(); 
  v_deriv->reform(class_num); 
  int cx; 
  double *deriv = v_deriv->point_u(); 
  /*---  multi-class logstic  ---*/
  if (class_num > 1 && loss_type == AzsLoss_Logistic) {
    AzDvect v_prob; 
    get_prob(v_pred, &v_prob); 
    const double *prob = v_prob.point(); 
    for (cx = 0; cx < class_num; ++cx) {
      deriv[cx] = (cx==cls) ? (prob[cx]-1) : prob[cx]; 
    }  
  }
  /*---  binary or one vs all  ---*/
  else {
    for (cx = 0; cx < class_num; ++cx) {
      double p = v_pred->get(cx); 
      double y = (cls == cx) ? 1 : -1; 
      deriv[cx] = getLossd(loss_type, p, y); 
    }
  }
}        

/*--------------------------------------------------------*/
/* static */
double AzsLmod::getLoss_classif(AzsLossType loss_type, 
                      const AzDvect *v_pred, /* predictions */
                      int cls) /* true class */
{
  int class_num = v_pred->rowNum(); 
  double loss = 0;  
  /*---  multi-class logistic  ---*/
  if (class_num > 1 && loss_type == AzsLoss_Logistic) {
    AzDvect v_prob; 
    get_prob(v_pred, &v_prob); 
    loss = -log(v_prob.get(cls)); 
  }
  /*---  binary or one vs all  ---*/
  else {
    int cx; 
    for (cx = 0; cx < class_num; ++cx) {
      double p = v_pred->get(cx); 
      double y = (cls == cx) ? 1 : -1; 
      loss += getLoss(loss_type, p, y); 
    }
  }
  return loss; 
}  

/*--------------------------------------------------------*/
#define my_exp(x) exp(MAX(-500,MIN(500,(x))))
/*------------------------------------------------------------------*/
/* static */
void AzsLmod::get_prob(const AzDvect *v_pred, /* input: predictions */
                       AzDvect *v_prob) /* output: probabilities */
{                  
  int num = v_pred->rowNum();      
  v_prob->reform(num); 
  const double *pred = v_pred->point(); 
  double *prob = v_prob->point_u(); 
  int cx; 
  for (cx = 0; cx < num; ++cx) {
    prob[cx] = my_exp(pred[cx]); 
  }
  v_prob->normalize1(); /* divide by sum */
}

/*------------------------------------------------------------------*/
/* static: return loss derivative */
double AzsLmod::getLossd(AzsLossType loss_type, 
                         double p, double y)
{
  double lossd = 0; 
  if (loss_type == AzsLoss_Square) {
    lossd = p-y; 
  }
  else if (loss_type == AzsLoss_Logistic) {  
    double py = p*y; 
    double ee = my_exp(-py);  
    lossd = -y*ee/(1+ee); 
  }
  else {
    throw new AzException("AzsLmod::getLossd", "unsupported loss type"); 
  }
  return lossd; 
}

/*--------------------------------------------------------*/
/* static: return loss */
double AzsLmod::getLoss(AzsLossType loss_type, 
                        double p, double y)                        
{
  double loss = 0; 
  if (loss_type == AzsLoss_Square) {
    double r = y-p; /* residual */
    loss = r*r/2; 
  }
  else if (loss_type == AzsLoss_Logistic) {
    double ee = my_exp(-p*y); 
    loss = log(1+ee); 
  }
  else {
    throw new AzException("AzsLmod::getLoss", "unsupported loss type"); 
  }
  return loss; 
}

/*------------------------------------------------------------------*/
void AzsLmod::allLoss_str(AzBytArr &s) {
  int ix; 
  for (ix = 0; ix < AzsLossType_Num; ++ix) {
    AzsLossType loss_type = (AzsLossType)ix;
    if (loss_type != AzsLoss_None) {   
      if (ix > 0) s.c(" | "); 
      s.c(lossName(loss_type)); 
    }    
  }
}

/*------------------------------------------------------------------*/
AzsLossType AzsLmod::lossType(const char *param) 
{
  AzBytArr str_param(param); 
  int ix; 
  for (ix = 0; ix < AzsLossType_Num; ++ix) {
    AzsLossType loss_type = (AzsLossType)ix; 
    if (str_param.compare(lossName(loss_type)) == 0) return loss_type;
  }
  return AzsLoss_None; 
}  

/*------------------------------------------------------------------*/
const char *AzsLmod::lossName(AzsLossType loss_type) 
{
  if (loss_type < 0 || loss_type >= AzsLossType_Num) {
    return "???"; 
  }
  return loss_str[loss_type]; 
}