'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Static.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Static.py" company="Terry D. Eppler">

     Boo

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the “Software”),
 to deal in the Software without restriction,
 including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software,
 and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

 You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

</copyright>
<summary>
	Static.py
</summary>
******************************************************************************************
'''
from enum import Enum, auto


class Client( Enum ):
	'''

		Purpose:
			Enumeration of auxiliary applications

	'''
	SQLite = auto( )
	Access = auto( )
	Excel = auto( )
	Word = auto( )
	Edge = auto( )
	Chrome = auto( )
	ControlPanel = auto( )
	Calculator = auto( )
	Outlook = auto( )
	Pyscripter = auto( )
	TaskManager = auto( )
	Storage = auto( )


class Scaler( Enum ):
	'''

		Enumeration of scaling algorythms

	'''
	Simple = auto( )
	Standard = auto( )
	Normal = auto( )
	OneHot = auto( )
	Neighbor = auto( )
	MinMax = auto( )
	Ordinal = auto( )
	Robust = auto( )