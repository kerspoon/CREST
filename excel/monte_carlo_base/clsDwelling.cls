Attribute VB_Name = "clsDwelling"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False

Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer
Private intResidents As Integer
Private intBuildingIndex As Integer
Private intPrimaryHeatingIndex As Integer
Private intPVSystemIndex As Integer
    
' ===============================================================================================
' Class Subroutines
    
' ===============================================================================================
' ## InitialiseDwelling
' Initialise the dwelling object
' ===============================================================================================
Public Sub InitialiseDwelling(index As Integer)
    
    Dim intOffset As Integer
    intOffset = 4
    
    ' // Get dwelling characteristics
    
    intDwellingIndex = index
    
    With wsDwellings
        intResidents = Application.index(.Range("rNumberResidentsIndex"), intDwellingIndex + intOffset).Value
        intBuildingIndex = Application.index(.Range("rBuildingIndex"), intDwellingIndex + intOffset).Value
        intPrimaryHeatingIndex = Application.index(.Range("rPrimaryHeatingIndex"), intDwellingIndex + intOffset).Value
        intPVSystemIndex = Application.index(.Range("rPvSystemIndex"), intDwellingIndex + intOffset).Value
    End With

End Sub
    
    
    
    
' ===============================================================================================
' ## WriteDwellingIndex
' Write the output to the worksheet
' ===============================================================================================
Public Sub WriteDwellingIndex(currentDate As Date, dwellingIndexRowOffset As Long)

    Dim intOffset As Integer
    Dim intMinute As Integer
    Dim strStartDate As String
    
    strStartDate = "00:00"
    
    intOffset = 6
    With wsResultsDisaggregated
        For intMinute = 1 To 1440
            .Range("A" + CStr(intMinute + intOffset)).Offset(dwellingIndexRowOffset, 0) = intDwellingIndex
            .Range("B" + CStr(intMinute + intOffset)).Offset(dwellingIndexRowOffset, 0) = currentDate
            .Range("C" + CStr(intMinute + intOffset)).Offset(dwellingIndexRowOffset, 0) = DateAdd("n", intMinute - 1, strStartDate)
        Next intMinute
    End With
End Sub


    
