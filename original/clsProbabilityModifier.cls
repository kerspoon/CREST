VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsProbabilityModifier"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit

Public IsWeekend As Boolean
Public ActiveOccupantCount As Integer
Public ID As String
Dim dModifiers(0 To 143) As Double

Public Property Get Modifiers(ByVal i As Integer) As Double
   Modifiers = dModifiers(i)
End Property

Public Property Let Modifiers(ByVal i As Integer, ByVal dProbability As Double)
   dModifiers(i) = dProbability
End Property

