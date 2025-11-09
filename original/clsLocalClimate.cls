VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsLocalClimate"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
' ===============================================================================================
' ## Local climate model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer

Private aClearnessIndex() As Double

' Outside temperature
Private aTheta_o() As Double

' An array to store the clear sky beam radiation at surface plane tracking sun (W/m^2)
Private aG_oClearsky() As Double

' global horizontal radiation (includes cloud cover) (W/m^2)
Private aG_o() As Double


' ===============================================================================================
' Class Properties

Public Property Get GetTheta_o(timestep As Integer)
    GetTheta_o = aTheta_o(timestep, 1)
End Property

Public Property Get GetG_o(timestep As Integer)
    GetG_o = aG_o(timestep, 1)
End Property

Public Property Get GetClearnessIndex(timestep As Integer)
    GetClearnessIndex = aClearnessIndex(timestep, 1)
End Property

Public Property Get GetG_oClearsky(timestep As Integer)
    GetG_oClearsky = aG_oClearsky(timestep, 1)
End Property


' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseLocalClimate
'
' ===============================================================================================
Public Sub InitialiseLocalClimate(dwellingIndex As Integer)

    intDwellingIndex = dwellingIndex
    aTheta_o() = objGlobalClimate.GetTheta_o
    aG_o() = objGlobalClimate.GetG_o
    aClearnessIndex() = objGlobalClimate.GetClearnessIndex
    aG_oClearsky() = objGlobalClimate.GetG_oClearsky

End Sub




' ===============================================================================================
' ## WriteLocalClimate
'
' ===============================================================================================
Public Sub WriteLocalClimate(dwellingIndexRowOffset As Long)
    With wsResultsDisaggregated
        .Range("I7:I1446").Offset(dwellingIndexRowOffset, 0) = aTheta_o
        .Range("J7:J1446").Offset(dwellingIndexRowOffset, 0) = aG_o
    End With
End Sub
