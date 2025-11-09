VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsLighting"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
' ===============================================================================================
' ## Domestic Lighting Model
' ===============================================================================================

Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer
Private intRunNumber As Integer

Private intResidents As Integer

Private intIrradianceThreshold As Integer

Private intBulbConfiguration As Integer
Private intNumBulbs As Integer

Private dblCalibrationScalar As Double

Private aSimulationArray(1 To 1443, 1 To 60) As Variant ' Declare an array to store the simulation data
' vSimulation array     (1,n)       Bulb number header
'                       (2,n)       Rating
'                       (3,n)       Relative use
'                       (4-1443,n)  Demand

Private aBulbArray As Variant ' Declare an array to store the lighting unit configuration
' aBulbArray            (1,1)   Example dwelling number
'                       (1,2)   Number of lighting units
'                       (1,3..)   Lighting unit ratings

Private aTotalLightingDemand(1 To 1440, 1 To 1) As Double

Private intActiveOccupancy() As Integer


' ===============================================================================================
' Class Properties

Public Property Get GetTotalLightingDemand(timestep As Integer)
    GetTotalLightingDemand = aTotalLightingDemand(timestep, 1)
End Property

Public Property Get ThermalGains() As Double()
    ' // Thermal gains for lighting is equal to the lighting demand
    ThermalGains = aTotalLightingDemand()
End Property

' // get the casual thermal gains from lighting for this timestep
Public Property Get GetPhi_cLighting(timestep As Integer)
    GetPhi_cLighting = aTotalLightingDemand(timestep, 1)
End Property
Public Property Get GetDailySumLighting() As Double
    GetDailySumLighting = WorksheetFunction.Sum(aTotalLightingDemand)
End Property

' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseLighting
'
' ===============================================================================================
Public Sub InitialiseLighting(dwellingIndex As Integer, runNumber As Integer)

    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    ' // Initialise the random number generator
    Randomize
    
    ' // Determine the irradiance threshold of this house
    With wsLightConfig
        intIrradianceThreshold = GetMonteCarloNormalDistGuess(.Range("iIrradianceThresholdMean").Value, .Range("iIrradianceThresholdSd").Value)
    End With
    
    ' // Choose a random house from the list of 100 provided in the bulbs sheet
    intBulbConfiguration = Int((100 * Rnd) + 1)
    
    ' // Get the bulb data
    aBulbArray = wsBulbs.Range("A" + CStr(intBulbConfiguration + 10) + ":BI" + CStr(intBulbConfiguration + 10))
    
    ' // Get the number of bulbs
    intNumBulbs = aBulbArray(1, 2)
    
    ' // Load the active occupancy array for this dwelling
    
    intActiveOccupancy = aOccupancy(intRunNumber).GetActiveOccupancy
        
    ' // Get the calibration scalar
    dblCalibrationScalar = wsLightConfig.Range("F24").Value
        
End Sub




' ===============================================================================================
' ## RunLightingSimulation
'
' ===============================================================================================
Public Sub RunLightingSimulation()
    
    ' // Define variables
    Dim i As Integer
    Dim j As Integer
    Dim strCol As String
    Dim r1 As Double
    Dim r2 As Double
    Dim cml As Double
    Dim intLightDuration As Integer
    Dim intLowerDuration As Integer
    Dim intUpperDuration As Integer
    Dim intRating As Integer
    Dim dblCalibratedRelativeUseWeighting As Double
    Dim intTime As Integer
    Dim intIrradiance As Integer
    Dim intActiveOccupants As Integer
    Dim dblEffectiveOccupancy As Double
    
    ' // For each bulb
    For i = 1 To intNumBulbs
        
        ' // Get the bulb rating
        intRating = aBulbArray(1, i + 2)
        
        'Scale down bulb power for Indian total lighting electricity demand
        blnIndia = IIf(wsMain.Range("rCountry").Value = "India", True, False)
        If blnIndia Then
            intRating = intRating * wsLightConfig.Range("G24").Value
        End If
            

        ' // Store the bulb number
        aSimulationArray(1, i) = i
        
        ' // Store the rating
        aSimulationArray(2, i) = intRating
                
        ' // Assign a random bulb use weighting to this bulb
        ' // Note that the calibration scalar is multiplied here to save processing time later
        dblCalibratedRelativeUseWeighting = -dblCalibrationScalar * Application.WorksheetFunction.Ln(Rnd())
        aSimulationArray(3, i) = dblCalibratedRelativeUseWeighting
        
        ' // Calculate the bulb usage at each minute of the day
        intTime = 1
        Do While (intTime <= 1440)
        
            ' // Is this bulb switched on to start with?
            ' // This concept is not implemented in this example.
            ' // The simplified assumption is that all bulbs are off to start with.
            
            ' // Get the irradiance for this minute
            intIrradiance = aLocalClimate(intRunNumber).GetG_o(intTime)
            
            ' // Get the number of current active occupants for this minute
            ' // Convert from 10 minute to 1 minute resolution
            intActiveOccupants = intActiveOccupancy(((intTime - 1) \ 10), 0)
                          
            ' // Determine if the bulb switch-on condition is passed
            ' // ie. Insuffient irradiance and at least one active occupant
            ' // There is a 5% chance of switch on event if the irradiance is above the threshold
            Dim blnLowIrradiance As Boolean
            blnLowIrradiance = ((intIrradiance < intIrradianceThreshold) Or (Rnd() < 0.05))

            ' // Get the effective occupancy for this number of active occupants to allow for sharing
            dblEffectiveOccupancy = wsLightConfig.Range("E" + CStr(37 + intActiveOccupants)).Value

            ' // Check the probability of a switch on at this time
            If (blnLowIrradiance And (Rnd() < (dblEffectiveOccupancy * dblCalibratedRelativeUseWeighting))) Then
            
                ' // This is a switch on event
            
                ' // Determine how long this bulb is on for
                r1 = Rnd()
                cml = 0
    
                For j = 1 To 9
                
                    ' // Get the cumulative probability of this duration
                    cml = wsLightConfig.Range("E" + CStr(54 + j)).Value
                
                    ' // Check to see if this is the type of light
                    If r1 < cml Then
                    
                        ' // Get the durations
                        intLowerDuration = wsLightConfig.Range("C" + CStr(54 + j)).Value
                        intUpperDuration = wsLightConfig.Range("D" + CStr(54 + j)).Value
                        
                        ' // Get another random number
                        r2 = Rnd()
                        
                        ' // Guess a duration in this range
                        intLightDuration = (r2 * (intUpperDuration - intLowerDuration)) + intLowerDuration
                                                  
                        ' // Exit the loop
                        Exit For
                    
                    End If
                Next j
                 
                For j = 1 To intLightDuration
                
                    ' // Range check
                    If intTime > 1440 Then Exit For
                    
                    ' // Get the number of current active occupants for this minute
                    intActiveOccupants = intActiveOccupancy(((intTime - 1) \ 10), 0)
                    
                    ' // If there are no active occupants, turn off the light
                    If intActiveOccupants = 0 Then Exit For
                    
                    ' // Store the demand
                    aSimulationArray(3 + intTime, i) = intRating
                        
                    ' // Increment the time
                    intTime = intTime + 1
                    
                Next j
                 
                 
            Else
                ' // The bulb remains off
                aSimulationArray(3 + intTime, i) = 0
                
                ' // Increment the time
                intTime = intTime + 1
                
            End If
        
        Loop
    
    Next i

End Sub



' ===============================================================================================
' ## TotalLightingDemand
'
' ===============================================================================================
Public Sub TotalLightingDemand()
    Dim intRow As Integer
    Dim intCol As Integer
    Dim dblRowSum As Double
           
    For intRow = 1 To 1440
    
        dblRowSum = 0
        
        For intCol = 1 To 60
        
            dblRowSum = dblRowSum + aSimulationArray(intRow + 3, intCol)
            
        Next intCol
        
        aTotalLightingDemand(intRow, 1) = dblRowSum
        
    Next intRow

    
End Sub




' ===============================================================================================
' ## WriteLighting
'
' ===============================================================================================
Public Sub WriteLighting(dwellingIndexRowOffset As Long)
    
    Dim intOffset As Integer
    
    intOffset = 6

    With wsResultsDisaggregated.Range("F" + CStr(1 + intOffset) + ":F" + CStr(1440 + intOffset))
        .Offset(dwellingIndexRowOffset, 0) = aTotalLightingDemand
    End With
    
End Sub
