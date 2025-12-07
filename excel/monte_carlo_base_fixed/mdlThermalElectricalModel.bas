Attribute VB_Name = "mdlThermalElectricalModel"
' ===============================================================================================
' ## CREST Demand Model v2.2
' ## A high-resolution stochastic integrated thermal-electrical domestic demand simulation tool
' ===============================================================================================
'
'    Copyright (C) 2017 John Barton and Murray Thomson
'    Centre for Renewable Energy Systems Technology (CREST),
'    Wolfson School of Mechanical, Electrical and Manufacturing Engineering
'    Loughborough University, Leicestershire LE11 3TU, UK
'    Tel. +44 1509 635350. Email address: J.P.Barton@lboro.ac.uk
'    Tel. +44 1509 635344. Email address: M.Thomson@lboro.ac.uk
'
'    This program is free software: you can redistribute it and/or modify
'    it under the terms of the GNU General Public License as published by
'    the Free Software Foundation, either version 3 of the License, or
'    (at your option) any later version.

'    This program is distributed in the hope that it will be useful,
'    but WITHOUT ANY WARRANTY; without even the implied warranty of
'    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
'    GNU General Public License for more details.

'    You should have received a copy of the GNU General Public License
'    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Option Explicit

' ===============================================================================================
' Code Module Variables

' // declare constants

Public Const PI As Double = 3.14159265359

' Specific heat capacity of water J/kg/K
Public Const SPECIFIC_HEAT_CAPACITY_WATER As Double = 4200

' // declare variables
' the date for the simulation run
Private dteDate As Date

' dwelling object array
Public aDwelling() As clsDwelling

' An array of occupancy objects
Public aOccupancy() As clsOccupancy

' An array of lighting objects
Public aLighting() As clsLighting

' An array of appliances objects
Public aAppliances() As clsAppliances

Public aBuilding() As clsBuilding

Public aPrimaryHeatingSystem() As clsHeatingSystem

Public aCoolingSystem() As clsCoolingSystem

Public objGlobalClimate As clsGlobalClimate

Public aLocalClimate() As clsLocalClimate

Public aHotWater() As clsHotWater

Public aHeatingControls() As clsHeatingControls

Public aPVSystem() As clsPVSystem

Public aSolarThermal() As clsSolarThermal

' a collection to store the activity statisitics
Public objActivityStatistics As Collection

' Declare an object to store activity statistics
Public objActivityStatsItem As clsProbabilityModifier

' The number of dwellings to simulate
Private intDwellingNumber As Integer
Private intTotalNumberSimulationRuns As Integer

' Row offsets for writing data to worksheets in case existing data is not to be overwritten
Public lngRowOffsetDwellings As Long
Public lngRowOffsetResultsDailyTotals As Long
Public lngRowOffsetResultsDisaggregated As Long
Public lngRowOffsetResultsAggregated As Long

' worksheet objects to fully qualify range references
Public wsMain As Worksheet
Public wsResultsDisaggregated As Worksheet
Public wsResultsAggregated As Worksheet
Public wsDwellings As Worksheet
Public wsGlobalClimate As Worksheet
Public wsClimateData As Worksheet
Public wsPVSystems As Worksheet
Public wsHeatingControls As Worksheet
Public wsPrimaryHeatingSystems As Worksheet
Public wsCoolingSystems As Worksheet
Public wsBuildings As Worksheet
Public wsClearnessIndexTPM As Worksheet
Public wsAppliancesAndWaterFixtures As Worksheet
Public wsActivityStats As Worksheet
Public wsLightConfig As Worksheet
Public wsBulbs As Worksheet
Public wsStartingStates As Worksheet
Public ws24hrOccupancy As Worksheet
Public wsWaterUsage As Worksheet
Public wsResultsDailySums As Worksheet
Public wsSolarThermal As Worksheet

' Variables to specify the location and year for appliance ownership,
' lighting level, heating system type and cooling system type (if any)
' Public blnIndia As Boolean
Public intYear As Integer      ' Year for appliance ownership, from 2006 to 2031
Public intWhole As Integer     ' Year rounded down to the nearest 5 years since 2005
Public intRemainder As Integer ' Interpolation index within 5 year intervals: a number between 0 and 4
Public intIndex1 As Integer    ' First or lower year index of proportion database
Public intIndex2 As Integer    ' Second or upper year index of proportion database
Public blnUK As Boolean        ' UK data for appliance ownership (does not change with year or urbanisation)
Public blnIndia As Boolean     ' India data for appliance ownership
Public blnUrban As Boolean     ' Urban data for appliance ownership
Public blnRural As Boolean     ' Urban data for appliance ownership

' ===============================================================================================
' Code Module Subroutines

' ===============================================================================================
' ## RunThermalElectricalDemandModel
'
' ===============================================================================================
Sub RunThermalElectricalDemandModel()

    ' // Declare variables
    Dim intDwellingIndex As Integer
    Dim intRunNumber As Integer
    Dim lngDwellingIndexRowOffset As Long
    Dim lngMaxDwellingIndex As Long
    Dim strErrorMessage As String
    
    ' a variable to store a reference to a range for the last cell in a worksheet
    Dim rngLastCell As Range
    
    ' Variables to determine the date for the simulation run
    Dim intDayOfMonth As Integer
    Dim intMonthOfYear As Integer

    ' // worksheet objects to fully qualify cell and range references
    Set wsMain = ThisWorkbook.Sheets("Main Sheet")
    Set wsResultsDisaggregated = ThisWorkbook.Sheets("Results - disaggregated")
    Set wsResultsAggregated = ThisWorkbook.Sheets("Results - aggregated")
    Set wsDwellings = ThisWorkbook.Sheets("Dwellings")
    Set wsGlobalClimate = ThisWorkbook.Sheets("GlobalClimate")
    Set wsClimateData = ThisWorkbook.Sheets("ClimateData&CoolingTech")
    Set wsPVSystems = ThisWorkbook.Sheets("PV Systems")
    Set wsHeatingControls = ThisWorkbook.Sheets("HeatingControls")
    Set wsPrimaryHeatingSystems = ThisWorkbook.Sheets("PrimaryHeatingSystems")
    Set wsCoolingSystems = ThisWorkbook.Sheets("CoolingSystems")
    Set wsBuildings = ThisWorkbook.Sheets("Buildings")
    Set wsClearnessIndexTPM = ThisWorkbook.Sheets("ClearnessIndexTPM")
    Set wsAppliancesAndWaterFixtures = ThisWorkbook.Sheets("AppliancesAndWaterFixtures")
    Set wsActivityStats = ThisWorkbook.Sheets("ActivityStats")
    Set wsLightConfig = ThisWorkbook.Sheets("light_config")
    Set wsBulbs = ThisWorkbook.Sheets("bulbs")
    Set wsStartingStates = ThisWorkbook.Sheets("Starting States")
    Set ws24hrOccupancy = ThisWorkbook.Sheets("24hr occupancy")
    Set wsWaterUsage = ThisWorkbook.Sheets("WaterUsage")
    Set wsResultsDailySums = ThisWorkbook.Sheets("Results - daily totals")
    Set wsSolarThermal = ThisWorkbook.Sheets("SolarThermalSystems")
    
    ' // Get the date
    intDayOfMonth = wsMain.Range("rDayOfMonth").Value
    intMonthOfYear = wsMain.Range("rMonthOfYear").Value
    dteDate = DateValue(CStr(intDayOfMonth) + "/" + CStr(intMonthOfYear) + "/2015")
    
    ' // Run the global climate model (to get irradiance and temperature shared by all dwellings)
    Set objGlobalClimate = New clsGlobalClimate
    objGlobalClimate.SimulateClearnessIndex
    objGlobalClimate.CalculateGlobalIrradiance
    objGlobalClimate.RunTemperatureModel
    ' // Now includes writing to worksheets
    
    ' // Write the global environmental variables to the worksheet
    ' objGlobalClimate.WriteGlobalClimate
    
    ' // Get number of simulation runs required
    intTotalNumberSimulationRuns = wsMain.Range("rSimulationRuns").Value
    
    ' // Reallocate the required amount of memory for the objects
    ReDim aDwelling(1 To intTotalNumberSimulationRuns)
    ReDim aOccupancy(1 To intTotalNumberSimulationRuns)
    ReDim aLighting(1 To intTotalNumberSimulationRuns)
    ReDim aAppliances(1 To intTotalNumberSimulationRuns)
    ReDim aBuilding(1 To intTotalNumberSimulationRuns)
    ReDim aPrimaryHeatingSystem(1 To intTotalNumberSimulationRuns)
    ReDim aCoolingSystem(1 To intTotalNumberSimulationRuns)
    ReDim aLocalClimate(1 To intTotalNumberSimulationRuns)
    ReDim aHotWater(1 To intTotalNumberSimulationRuns)
    ReDim aHeatingControls(1 To intTotalNumberSimulationRuns)
    ReDim aPVSystem(1 To intTotalNumberSimulationRuns)
    ReDim aSolarThermal(1 To intTotalNumberSimulationRuns)
    
    ' // If existing data is to be overwritten then ...
    If wsMain.Shapes("objOverWriteData").ControlFormat.Value = 1 Then
        ' // First sense check the number of runs specified
        If (wsMain.Shapes("objDynamicOutput").ControlFormat.Value = 1) _
            And (intTotalNumberSimulationRuns >= 729) Then
            wsMain.Range("J18") = "Error - simulation stopped"
            ' // this will produce more rows of results than can fit into the worksheet
            MsgBox "Please enter a number of dwellings less than 729.", vbExclamation
            End
        End If
        
        ' // Clear the relevant worksheets of existing data
        wsResultsDisaggregated.Rows(7 & ":" & wsResultsDisaggregated.Rows.Count).Clear
        wsResultsAggregated.Rows(5 & ":" & wsResultsAggregated.Rows.Count).Clear
        wsResultsDailySums.Rows(5 & ":" & wsResultsDailySums.Rows.Count).Clear
        
        ' // ... only clear the Dwellings worksheet if the parameters are being assigned stochastically
        If wsMain.Shapes("objAssignDwellingParameters").ControlFormat.Value = 1 Then
            wsDwellings.Rows(5 & ":" & wsDwellings.Rows.Count).Clear
            
        End If
        
        ' // and set the row writing offsets to the defaults
        lngRowOffsetResultsDailyTotals = 4
        lngRowOffsetResultsDisaggregated = 6
        lngRowOffsetResultsAggregated = 4
        lngRowOffsetDwellings = 4
        
    ' // Otherwise find the last non-empty row the worksheets and set these as offsets
    ' // Except for the Dwellings worksheet if parameters are being assigned manually
    Else
        
        Set rngLastCell = wsResultsDailySums.Cells(wsResultsDailySums.Rows.Count, 1).End(xlUp)
        lngRowOffsetResultsDailyTotals = WorksheetFunction.Max(rngLastCell.Row, 4)
        
        Set rngLastCell = wsResultsDisaggregated.Cells(wsResultsDisaggregated.Rows.Count, 1).End(xlUp)
        lngRowOffsetResultsDisaggregated = WorksheetFunction.Max(rngLastCell.Row, 6)
        
        Set rngLastCell = wsResultsAggregated.Cells(wsResultsAggregated.Rows.Count, 1).End(xlUp)
        lngRowOffsetResultsAggregated = WorksheetFunction.Max(rngLastCell.Row, 4)
        
        If wsMain.Shapes("objAssignDwellingParameters").ControlFormat.Value = 1 Then
            Set rngLastCell = wsDwellings.Cells(wsDwellings.Rows.Count, 1).End(xlUp)
            lngRowOffsetDwellings = WorksheetFunction.Max(rngLastCell.Row, 4)
        Else
            strErrorMessage = "If dwelling parameters are manually assigned, existing data must be overwritten (input 8.)"
            MsgBox strErrorMessage, vbExclamation
        End If

            
        ' // Sense check the number of simulation runs specified
        lngMaxDwellingIndex = lngRowOffsetDwellings - 4 + intTotalNumberSimulationRuns
        If (wsMain.Shapes("objDynamicOutput").ControlFormat.Value = 1) _
            And (lngMaxDwellingIndex >= 729) Then
            strErrorMessage = "Please enter a number of dwellings less than " + CStr(729 - (lngRowOffsetDwellings - 4)) + "."
            ' // this will produce more rows of results than can fit into the worksheet
            wsMain.Range("J18") = "Error - simulation stopped"
            MsgBox strErrorMessage, vbExclamation
            End
        End If
    End If
    
    ' // Choose appliance ownership probabilities, appropriate to UK or India
    ' // If India is chosen, the ownership depends on year and 'urban' or 'rural'
    Call SetApplianceDatabase
    
    ' // Similarly, set the proportions of building types
    Call SetBuildingProportions
    
    ' // and similarly, set the proportions of heating systems
    Call SetHeatingSystemProportions

    ' // and similarly, set the proportions of cooling systems
    Call SetCoolingSystemProportions
    
    ' // Load the activity statistics into a collection
    ' // (note: the same activity probability profiles are used for each dwelling)
    LoadActivityStatistics
    
    For intRunNumber = 1 To intTotalNumberSimulationRuns
        ' // Check if the user has stopped the simulation
        DoEvents
        If wsMain.Range("J18") = "Wait ..." Then Exit Sub
        
        ' Run the climate model for each house, only if required to establish scalar factor
        ' objGlobalClimate.SimulateClearnessIndex
        ' objGlobalClimate.CalculateGlobalIrradiance
        ' objGlobalClimate.RunTemperatureModel
        
        ' // Set the dwelling index number
        intDwellingIndex = lngRowOffsetDwellings - 4 + intRunNumber
        
        ' // Check if dwelling parameters are to be assigned stochastically
        If wsMain.Shapes("objAssignDwellingParameters").ControlFormat.Value = 1 Then
            ' // then stochastically assign new dwelling parameters
            AssignDwellingParameters intDwellingIndex
        End If
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // DWELLINGS
         
        ' // Create object instance
        Set aDwelling(intRunNumber) = New clsDwelling
        
        ' // Initialise dwelling object
        aDwelling(intRunNumber).InitialiseDwelling (intDwellingIndex)
        
        ' '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // LOCAL CLIMATES
        
        ' // Create object
        Set aLocalClimate(intRunNumber) = New clsLocalClimate
        
        ' // Initialise object
        ' // Every dwelling's local climate is the same as the global climate
        aLocalClimate(intRunNumber).InitialiseLocalClimate intDwellingIndex
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // OCCUPANCY
        ' // Create object instances
        Set aOccupancy(intRunNumber) = New clsOccupancy
        
        ' // Initialise occupancy object
        aOccupancy(intRunNumber).InitialiseOccupancy (intDwellingIndex)
                
        ' // Run occupancy model
        aOccupancy(intRunNumber).RunFourStateOccupancySimulation
                
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // LIGHTING
        ' // Create object instance
        Set aLighting(intRunNumber) = New clsLighting
        
        ' // Initialise lighting object
        aLighting(intRunNumber).InitialiseLighting intDwellingIndex, intRunNumber
        
        ' // Run lighting model
        aLighting(intRunNumber).RunLightingSimulation
                
        ' // Calculate the total lighting demand
        aLighting(intRunNumber).TotalLightingDemand
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // APPLIANCES
        ' // Create object instance
        Set aAppliances(intRunNumber) = New clsAppliances
        
        ' // Initialise appliance object
        aAppliances(intRunNumber).InitialiseAppliances intDwellingIndex, intRunNumber
        
        ' // Run appliance model
        aAppliances(intRunNumber).RunApplianceSimulation
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // PV SYSTEM MODEL
        
        ' // Create object
        Set aPVSystem(intRunNumber) = New clsPVSystem
                
        ' // Initialise object
        aPVSystem(intRunNumber).InitialisePVSystem intDwellingIndex, intRunNumber
                        
        ' // Run PV model
        aPVSystem(intRunNumber).CalculatePVOutput
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // CASUAL THERMAL GAINS (OCCUPANCY, LIGHTING, APPLIANCES)
        
        ' // Calculate thermal gains from occupancy and appliances
        aOccupancy(intRunNumber).CalculateThermalGains
        aAppliances(intRunNumber).CalculateThermalGains
                                
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // PRIMARY HEATING SYSTEM
        
        ' // Create object
        Set aPrimaryHeatingSystem(intRunNumber) = New clsHeatingSystem
        
        ' // Initialise object
        aPrimaryHeatingSystem(intRunNumber).InitialiseHeatingSystem intDwellingIndex, intRunNumber
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // PRIMARY COOLING SYSTEM
        
        ' // Create object
        Set aCoolingSystem(intRunNumber) = New clsCoolingSystem
        
        ' // Initialise object
        aCoolingSystem(intRunNumber).InitialiseCoolingSystem intDwellingIndex, intRunNumber
           
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // BUILDING
        
        ' // Create object
        Set aBuilding(intRunNumber) = New clsBuilding
        
        ' // Initialise object
        aBuilding(intRunNumber).InitialiseBuilding intDwellingIndex, intRunNumber
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // HOT WATER
        
        ' // Create object
        Set aHotWater(intRunNumber) = New clsHotWater
        
        ' // Initialise object
        aHotWater(intRunNumber).InitialiseHotWater intDwellingIndex, intRunNumber
        
        ' // Calculate hot water demand
        aHotWater(intRunNumber).RunHotWaterDemandSimulation
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // HEATING CONTROLS
        
        ' // Create object
        Set aHeatingControls(intRunNumber) = New clsHeatingControls
        
        ' // Initialise object
        aHeatingControls(intRunNumber).InitialiseHeatingControls intDwellingIndex, intRunNumber
                       
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // SOLAR THERMAL
        
        ' // Create object
        Set aSolarThermal(intRunNumber) = New clsSolarThermal
        
        ' // Initialise object
        aSolarThermal(intRunNumber).InitialiseSolarThermal intDwellingIndex, intRunNumber
                       
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // RUN THERMAL DEMAND MODEL
        
        Dim intMinute As Integer
        
        For intMinute = 1 To 1440
            ' // Given temperature conditions, calculate heating system control variables
            aHeatingControls(intRunNumber).CalculateControlStates (intMinute)
                        
            ' // Calculate the heat output of the heating system (can be zero)
            aPrimaryHeatingSystem(intRunNumber).CalculateHeatOutput (intMinute)
            
            ' // Calculate the cooling output of the cooling system (can be zero)
            aCoolingSystem(intRunNumber).CalculateCoolingOutput (intMinute)
            
            ' // Calculate the heat output of the solar thermal collector (if any)
            aSolarThermal(intRunNumber).CalculateSolarThermalOutput (intMinute)
                        
            ' // Calculate the temperature changes in the dwelling (includes hot water)
            aBuilding(intRunNumber).CalculateTemperatureChange (intMinute)
            
        Next intMinute
                           
        ' // Calculate the total appliance demand (includes heating and cooling system power demands)
        aAppliances(intRunNumber).TotalApplianceDemand
        
        ' // Calculate the net demand
        aPVSystem(intRunNumber).CalculateNetDemand
        
        ' // Calculate the self consumption
        aPVSystem(intRunNumber).CalculateSelfConsumption
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' // WRITE RESULTS FOR INDIVIDUAL DWELLING
             
        ' // If specified, write detailed dynamic simulation output to results worksheet
        If wsMain.Shapes("objDynamicOutput").ControlFormat.Value = 1 Then
            lngDwellingIndexRowOffset = 1440 * (CLng(intDwellingIndex) - 1)
            aDwelling(intRunNumber).WriteDwellingIndex dteDate, lngDwellingIndexRowOffset
            aOccupancy(intRunNumber).WriteOccupancy lngDwellingIndexRowOffset
            aAppliances(intRunNumber).WriteAppliances lngDwellingIndexRowOffset
            aLighting(intRunNumber).WriteLighting lngDwellingIndexRowOffset
            aLocalClimate(intRunNumber).WriteLocalClimate lngDwellingIndexRowOffset
            aPrimaryHeatingSystem(intRunNumber).WriteHeatingSystem lngDwellingIndexRowOffset
            aCoolingSystem(intRunNumber).WriteCoolingSystem lngDwellingIndexRowOffset
            aBuilding(intRunNumber).WriteBuilding lngDwellingIndexRowOffset
            aHotWater(intRunNumber).WriteHotWater lngDwellingIndexRowOffset
            aHeatingControls(intRunNumber).WriteHeatingControls
            aPVSystem(intRunNumber).WritePVSystem lngDwellingIndexRowOffset
            aSolarThermal(intRunNumber).WriteSolarThermal lngDwellingIndexRowOffset
            
        End If
        
        ' // If specified, calculate and write sums of daily output for the dwelling
        If wsMain.Shapes("objDailyTotals").ControlFormat.Value = 1 Then
            DailyTotals intDwellingIndex, intRunNumber, dteDate
        End If
    Next intRunNumber
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' // CALCULATE AGGREGATE RESULTS FOR ALL DWELLINGS
    
    ' // If specified, calculate and write aggregated high-resolution dynamic results to worksheet
    If wsMain.Shapes("objDynamicOutput").ControlFormat.Value = 1 Then
        AggregateResults (intDwellingNumber)
    End If
    
End Sub




' ===============================================================================================
' ## RunSimulationButton_Click
'
' ===============================================================================================
Sub RunSimulationButton_Click()
    ' // Switch off auto-calculation
    Application.Calculation = xlCalculationManual
    
    ' // Write text to sheet
    ThisWorkbook.Worksheets("Main Sheet").Range("J18") = "Running"
'    Stop
    
    
    ' // Run the simulation
    Call RunThermalElectricalDemandModel
    
    ' // Switch on auto-calculation
    Application.Calculation = xlCalculationAutomatic
    
    ' // Re-calculate the sheet
    Application.Calculate
    
    ThisWorkbook.Worksheets("Main Sheet").Range("J18") = "Stopped"
    
End Sub




' ===============================================================================================
' ## StopSimulationButton_Click
' Routine to force the model to exit main subroutine
' ===============================================================================================
Sub StopSimulationButton_Click()
    
    If ThisWorkbook.Sheets("Main Sheet").Range("J18") = "Running" Then
        ThisWorkbook.Sheets("Main Sheet").Range("J18") = "Wait ..."
    End If
End Sub

' ===============================================================================================
' ## SetApplianceDatabase
' Changes with the country, the year and urban or rural location.
' Needed to set the database that the InitialiseAppliances subprogram draws from
' ===============================================================================================
Public Sub SetApplianceDatabase()

    ' // Appliance ownerships per dwelling
    Dim aProportionsDatabase() As Variant ' All columns to be read in
    Dim aEnergiesDatabase() As Variant ' All columns to be read in
    Dim aOperatingPowersDatabase() As Variant ' All columns to be read in
    Dim aStandbyPowersDatabase() As Variant ' All columns to be read in
    Dim aProportionColumn(1 To 31, 1 To 1) As Variant 'Column copied and pasted to spreadseet input column
    Dim aEnergiesColumn(1 To 31, 1 To 1) As Variant 'Column copied and pasted to spreadseet input column
    Dim aOperatingPowersColumn(1 To 31, 1 To 1) As Variant 'Column copied and pasted to spreadseet input column
    Dim aStandbyPowersColumn(1 To 31, 1 To 1) As Variant 'Column copied and pasted to spreadseet input column
    Dim i As Integer    ' Integer for looping
    Dim intProportionsOffset As Integer 'Column offset when choosing appliance ownership proportions
    ' Dim intEnergiesOffset As Integer 'Column offset when choosing appliance energy use and power data
    ' // Get the year
    intYear = wsMain.Range("rYear").Value
    ' intDayOfMonth = wsMain.Range("rDayOfMonth").Value
    intWhole = WorksheetFunction.Floor(intYear - 1, 5)
    'wsMain.Range("G1") = CStr(intWhole) 'Diagnostic only
    intRemainder = intYear - intWhole - 1
    'MsgBox (intWhole)
    'MsgBox (intRemainder)
    intIndex1 = (intWhole - 2005) / 5 + 1
    intIndex2 = intIndex1 - (intRemainder > 0) 'For some really strange reason, TRUE is interpreted as -1 !!
    'MsgBox (intIndex1)  'IntIndex1 and IntIndex2 are the indices of the columns between which
    'MsgBox (intIndex2)  'appliance ownership data is interpolated.
    ' // Get the country for appliance ownership
    blnUK = IIf(wsMain.Range("rCountry").Value = "UK", True, False)
    blnIndia = IIf(wsMain.Range("rCountry").Value = "India", True, False)
    ' // Get urban or rural location for appliance ownership
    blnUrban = IIf(wsMain.Range("rUrbanRural").Value = "Urban", True, False)
    blnRural = IIf(wsMain.Range("rUrbanRural").Value = "Rural", True, False)
    ' MsgBox ("Rural " + CStr(blnRural))
    ' Read in the appropriate column of appliance ownerships
    aProportionsDatabase = wsAppliancesAndWaterFixtures.Range("rProportionsDatabase")
    ' wsAppliancesAndWaterFixtures.Range("AV9:BH39") = aProportionsDatabase
    aEnergiesDatabase = wsAppliancesAndWaterFixtures.Range("rEnergiesDatabase")
    ' wsAppliancesAndWaterFixtures.Range("AV9:BH39") = aProportionsDatabase
    aOperatingPowersDatabase = wsAppliancesAndWaterFixtures.Range("rOperatingPowersDatabase")
    ' wsAppliancesAndWaterFixtures.Range("AV9:BH39") = aProportionsDatabase
    aStandbyPowersDatabase = wsAppliancesAndWaterFixtures.Range("rStandbyPowersDatabase")
    ' wsAppliancesAndWaterFixtures.Range("AV9:BH39") = aProportionsDatabase
    If blnUK Then
    'MsgBox ("UK " + CStr(blnUK))
        'MsgBox (aProportionsDatabase(1, 1))
        For i = 1 To 31
            aProportionColumn(i, 1) = aProportionsDatabase(i, 1) 'UK appliance ownership
            ' MsgBox (aProportionColumn(i, 1))
            aEnergiesColumn(i, 1) = aEnergiesDatabase(i, 1) 'UK energy per appliance
            aOperatingPowersColumn(i, 1) = aOperatingPowersDatabase(i, 1) 'UK operating powers, watts
            aStandbyPowersColumn(i, 1) = aStandbyPowersDatabase(i, 1) 'UK standby powers, watts
        Next i
        'MsgBox ("Check 1") 'used for debugging
    ElseIf blnIndia Then
        If blnUrban Then
            intProportionsOffset = 1
        ElseIf blnRural Then
            intProportionsOffset = 7
        Else
            wsMain.Range("J18") = "Error - simulation stopped"
            MsgBox "Please enter Rural or Urban", vbExclamation
            End
        End If
        'Choose column and transfer numbers
        For i = 1 To 31
            aProportionColumn(i, 1) = aProportionsDatabase(i, intProportionsOffset + intIndex1) * (5 - intRemainder) / 5 _
            + aProportionsDatabase(i, intProportionsOffset + intIndex2) * intRemainder / 5
            aEnergiesColumn(i, 1) = aEnergiesDatabase(i, 1 + intIndex1) * (5 - intRemainder) / 5 _
            + aEnergiesDatabase(i, 1 + intIndex2) * intRemainder / 5
            aOperatingPowersColumn(i, 1) = aOperatingPowersDatabase(i, 1 + intIndex1) * (5 - intRemainder) / 5 _
            + aOperatingPowersDatabase(i, 1 + intIndex2) * intRemainder / 5
            aStandbyPowersColumn(i, 1) = aStandbyPowersDatabase(i, 1 + intIndex1) * (5 - intRemainder) / 5 _
            + aStandbyPowersDatabase(i, 1 + intIndex2) * intRemainder / 5
        Next i
    Else
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "That country's appliance ownership is not yet known", vbExclamation
        End
    End If
    wsAppliancesAndWaterFixtures.Range("F9:F39") = aProportionColumn 'NB This format only works for 2D arrays.
    wsAppliancesAndWaterFixtures.Range("V9:V39") = aEnergiesColumn
    wsAppliancesAndWaterFixtures.Range("P9:P39") = aOperatingPowersColumn
    wsAppliancesAndWaterFixtures.Range("T9:T39") = aStandbyPowersColumn
    'With wsAppliancesAndWaterFixtures
    '    For i = 1 To 31
    '        .Range("F" + CStr(i + 8)) = aProportionColumn(i, 1)
    '    Next i
    'End With
End Sub
Public Sub SetBuildingProportions()

    ' // Appliance ownerships per dwelling
    Dim aBuildingDatabase() As Variant ' All columns to be read in
    Dim aBuildingColumn(1 To 8, 1 To 1) As Variant 'Column copied and pasted to spreadsheet input column
    Dim i As Integer    ' Integer for looping
    Dim intBuildingOffset As Integer 'Column offset when choosing heating system proportions
        
    ' Read in the appropriate column of heating system proportions
    aBuildingDatabase = wsBuildings.Range("rBuildingProportionDatabase")
    
    If blnUK Then
    'MsgBox ("UK " + CStr(blnUK))
        'MsgBox (aProportionsDatabase(1, 1))
        For i = 1 To 7
            aBuildingColumn(i, 1) = aBuildingDatabase(i, 1) 'UK appliance ownership
        Next i
    ElseIf blnIndia Then
        If blnUrban Then
            intBuildingOffset = 1
        ElseIf blnRural Then
            intBuildingOffset = 7
        Else
            wsMain.Range("J18") = "Error - simulation stopped"
            MsgBox "Please enter Rural or Urban", vbExclamation
            End
        End If
        'Choose column and transfer numbers
        For i = 1 To 8
            aBuildingColumn(i, 1) = aBuildingDatabase(i, intBuildingOffset + intIndex1) * (5 - intRemainder) / 5 _
            + aBuildingDatabase(i, intBuildingOffset + intIndex2) * intRemainder / 5
        Next i
    Else
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "That country's building types are not yet known", vbExclamation
        End
    End If
    wsBuildings.Range("B5:B12") = aBuildingColumn 'NB This format only works for 2D arrays.
End Sub

Public Sub SetHeatingSystemProportions()

    ' // Appliance ownerships per dwelling
    Dim aHeatingDatabase() As Variant ' All columns to be read in
    Dim aHeatingColumn(1 To 7, 1 To 1) As Variant 'Column copied and pasted to spreadsheet input column
    Dim i As Integer    ' Integer for looping
    Dim intHeatingOffset As Integer 'Column offset when choosing heating system proportions
    
    ' Read in the appropriate column of heating system proportions
    aHeatingDatabase = wsPrimaryHeatingSystems.Range("rHeatingProportionDatabase")
    
    If blnUK Then
    'MsgBox ("UK " + CStr(blnUK))
        'MsgBox (aProportionsDatabase(1, 1))
        For i = 1 To 5
            aHeatingColumn(i, 1) = aHeatingDatabase(i, 1) 'UK appliance ownership
        Next i
    ElseIf blnIndia Then
        If blnUrban Then
            intHeatingOffset = 1
        ElseIf blnRural Then
            intHeatingOffset = 7
        Else
            wsMain.Range("J18") = "Error - simulation stopped"
            MsgBox "Please enter Rural or Urban", vbExclamation
            End
        End If
        'Choose column and transfer numbers
        For i = 1 To 5
            aHeatingColumn(i, 1) = aHeatingDatabase(i, intHeatingOffset + intIndex1) * (5 - intRemainder) / 5 _
            + aHeatingDatabase(i, intHeatingOffset + intIndex2) * intRemainder / 5
        Next i
    Else
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "That country's appliance ownership is not yet known", vbExclamation
        End
    End If
    wsPrimaryHeatingSystems.Range("B5:B9") = aHeatingColumn 'NB This format only works for 2D arrays.
End Sub
Public Sub SetCoolingSystemProportions()

    ' // Appliance ownerships per dwelling
    Dim aCoolingDatabase() As Variant ' All columns to be read in
    Dim aCoolingColumn(1 To 7, 1 To 1) As Variant 'Column copied and pasted to spreadsheet input column
    Dim i As Integer    ' Integer for looping
    Dim intCoolingOffset As Integer 'Column offset when choosing Cooling system proportions
    
    ' Read in the appropriate column of Cooling system proportions
    aCoolingDatabase = wsCoolingSystems.Range("rCoolingProportionDatabase")
    
    If blnUK Then
    'MsgBox ("UK " + CStr(blnUK))
        'MsgBox (aProportionsDatabase(1, 1))
        For i = 1 To 4
            aCoolingColumn(i, 1) = aCoolingDatabase(i, 1) 'UK cooling ownership
        Next i
    ElseIf blnIndia Then
        If blnUrban Then
            intCoolingOffset = 1
        ElseIf blnRural Then
            intCoolingOffset = 7
        Else
            wsMain.Range("J18") = "Error - simulation stopped"
            MsgBox "Please enter Rural or Urban", vbExclamation
            End
        End If
        'Choose column and transfer numbers
        For i = 1 To 4
            aCoolingColumn(i, 1) = aCoolingDatabase(i, intCoolingOffset + intIndex1) * (5 - intRemainder) / 5 _
            + aCoolingDatabase(i, intCoolingOffset + intIndex2) * intRemainder / 5
        Next i
    Else
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "That country's appliance ownership is not yet known", vbExclamation
        End
    End If
    wsCoolingSystems.Range("B5:B8") = aCoolingColumn 'NB This format only works for 2D arrays.
End Sub


' ===============================================================================================
' ## LoadActivityStatistics
' Load the activity statistics into a collection (needed for the appliance and hot water demand model)
' ===============================================================================================
Private Sub LoadActivityStatistics()

    ' // Declare the variables
    Dim i, j As Integer
    Dim strKey As String
    Dim strCell As String
    

    Set objActivityStatistics = New Collection
    
    ' // Load the activity statistics into probability modifiers objects
    For i = 30 To 101
    
        ' // Create a new probability modifier
        Set objActivityStatsItem = New clsProbabilityModifier
    
        ' // Read in the data
        objActivityStatsItem.IsWeekend = IIf(wsActivityStats.Range("B" + CStr(i)).Value = 1, True, False)
        objActivityStatsItem.ActiveOccupantCount = wsActivityStats.Range("C" + CStr(i)).Value
        objActivityStatsItem.ID = wsActivityStats.Range("D" + CStr(i)).Value
        
        ' // Get the hourly modifiers
        For j = 0 To 143
        
            ' // Get the column reference
            strCell = Cells(i, j + 5).Address(True, False, xlA1)
        
            ' // Read the values
            objActivityStatsItem.Modifiers(j) = wsActivityStats.Range(strCell).Value
            
        Next j

        ' // Now generate a key
        strKey = IIf(objActivityStatsItem.IsWeekend, "1", "0") + "_" + CStr(objActivityStatsItem.ActiveOccupantCount) + "_" + objActivityStatsItem.ID
        
        ' // Add this object to the collection
        objActivityStatistics.Add Item:=objActivityStatsItem, Key:=strKey
    
    Next i

End Sub




' ===============================================================================================
' ## AggregateResults
'
' ===============================================================================================
Public Sub AggregateResults(iNumber As Integer)
    Dim intDwellingNumber As Integer
    
    Dim intMinute As Integer
    
    Dim intDwellingIndex As Integer
    Dim lngMaxDwellingIndex As Long
    Dim lngCurrentDwellingIndex As Long
    
    ' A variable to store a column reference
    Dim intCol As Integer
    
    Dim intOffset As Integer
    
    Dim intTotalPopulation As Integer
    
    Dim dblCumulativeOccupancy As Double
    
    Dim dblCumulativeActivity As Double
    
    ' total electricity demand for dwellings (appliances and lighting)
    Dim dblP_e As Double
    
    ' total PV output for all dwellings
    Dim dblP_pv As Double
    
    ' total net electricity demand for all dwellings
    Dim dblP_net As Double
    
    ' space heating timer state
    Dim blnSpaceTimer As Integer
    
    ' space heating thermostat state
    Dim blnSpaceThermo As Integer
    
    ' hot water timer state
    Dim blnWaterTimer As Integer
    
    ' hot water thermostat state
    Dim blnWaterThermo As Integer
    
    ' a variable to store the thermal output from a heating system
    Dim dblPhi_h As Double
    
    ' total thermal demand for space heating
    Dim dblPhi_hSpace As Double
    
    ' total thermal demand for hot water
    Dim dblPhi_hWater As Double
    
    Dim dblPhi_s As Double
    
    Dim dblTheta_i As Double
    
    Dim dblV_dhw As Double
    
    Dim dblTheta_cyl As Double
    
    Dim dblSpaceThermostatState As Double
    
    Dim dblSpaceTimerState As Double
    
    Dim dblHotWaterThermostatState As Double
    
    Dim dblHotWaterTimerState As Double
    
    Dim dblHeatingOnOff As Double
    
    Dim dblHotWaterHeatingOnOff As Double
    
    Dim dblM_fuel As Double
    
    Dim dblPhi_collector As Double
    
    Dim dblP_self As Double
    
    Dim dblCoolingTimerState As Double
    
    Dim dblCoolingOnOff As Double
        
    ' // arrays to hold the aggregated time series values
    Dim aOccupancy(1 To 1440, 1 To 1) As Double
    
    Dim aActivity(1 To 1440, 1 To 1) As Double
    
    Dim aP_e(1 To 1440, 1 To 1) As Double
    
    Dim aP_pv(1 To 1440, 1 To 1) As Double
    
    Dim aP_net(1 To 1440, 1 To 1) As Double
    
    Dim aPhi_hSpace(1 To 1440, 1 To 1) As Double
    
    Dim aPhi_hWater(1 To 1440, 1 To 1) As Double
    
    Dim aPhi_s(1 To 1440, 1 To 1) As Double
    
    Dim aTheta_i(1 To 1440, 1 To 1) As Double
    
    Dim aV_dhw(1 To 1440, 1 To 1) As Double
    
    Dim aTheta_cyl(1 To 1440, 1 To 1) As Double
    
    Dim aSpaceTimerState(1 To 1440, 1 To 1) As Double
    
    Dim aHotWaterTimerState(1 To 1440, 1 To 1) As Double
    
    Dim aHeatingOnOff(1 To 1440, 1 To 1) As Double
    
    Dim aHotWaterHeatingOnOff(1 To 1440, 1 To 1) As Double
    
    Dim aM_fuel(1 To 1440, 1 To 1) As Double
    
    Dim aPhi_collector(1 To 1440, 1 To 1) As Double
    
    Dim aP_self(1 To 1440, 1 To 1) As Double
    
    Dim aMeanV_dhw(1 To 1440, 1 To 1) As Double
    
    Dim aMeanV_gas(1 To 1440, 1 To 1) As Double
    
    Dim aCoolingTimerState(1 To 1440, 1 To 1) As Double
    
    Dim aCoolingOnOff(1 To 1440, 1 To 1) As Double
    
    intDwellingNumber = iNumber
    lngMaxDwellingIndex = lngRowOffsetDwellings - 4 + intTotalNumberSimulationRuns
    intOffset = 6
          
    ' // Get total population to normalise occupancy and activity
    intTotalPopulation = WorksheetFunction.Sum(wsDwellings.Range("B5:B" + CStr(4 + lngMaxDwellingIndex)))
    
    ' // Loop through the minutes of the day
    For intMinute = 1 To 1440
        
        ' // Set the time
        wsResultsAggregated.Cells(4 + intMinute, 2) = wsResultsDisaggregated.Cells(6 + intMinute, 3)
        
        ' // Set the variables for this minute to zero
        dblCumulativeOccupancy = 0
        dblCumulativeActivity = 0
        dblP_e = 0
        dblP_pv = 0
        dblPhi_hSpace = 0
        dblPhi_hWater = 0
        dblPhi_s = 0
        dblTheta_i = 0
        dblV_dhw = 0
        dblTheta_cyl = 0
        dblSpaceTimerState = 0
        dblHotWaterTimerState = 0
        dblHeatingOnOff = 0
        dblHotWaterHeatingOnOff = 0
        dblM_fuel = 0
        dblPhi_collector = 0
        dblP_self = 0
        dblCoolingTimerState = 0
        dblCoolingOnOff = 0
        
        ' // Loop through the dwellings
        For lngCurrentDwellingIndex = 1 To lngMaxDwellingIndex
            
            With wsResultsDisaggregated
                
                ' // Get the dwelling's contribution to the aggregated total
                dblCumulativeOccupancy = dblCumulativeOccupancy + .Cells(intOffset + intMinute + 1440 * (lngCurrentDwellingIndex - 1), 4).Value
                dblCumulativeActivity = dblCumulativeActivity + .Cells(intOffset + intMinute + 1440 * (lngCurrentDwellingIndex - 1), 5).Value
                dblP_e = dblP_e + .Cells(intOffset + intMinute + 1440 * (lngCurrentDwellingIndex - 1), 6).Value + .Cells(intOffset + intMinute + 1440 * (lngCurrentDwellingIndex - 1), 7).Value
                dblP_pv = dblP_pv + .Range("W6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblPhi_hWater = dblPhi_hWater + .Range("Z6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblPhi_hSpace = dblPhi_hSpace + .Range("Y6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblPhi_s = dblPhi_s + .Range("K6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblTheta_i = dblTheta_i + .Range("N6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblV_dhw = dblV_dhw + .Range("O6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblTheta_cyl = dblTheta_cyl + .Range("P6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblSpaceTimerState = dblSpaceTimerState + .Range("Q6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblHotWaterTimerState = dblHotWaterTimerState + .Range("R6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblHeatingOnOff = dblHeatingOnOff + .Range("S6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblHotWaterHeatingOnOff = dblHotWaterHeatingOnOff + .Range("T6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblM_fuel = dblM_fuel + .Range("AA6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblPhi_collector = dblPhi_collector + .Range("AE6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblP_self = dblP_self + .Range("AF6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblCoolingTimerState = dblCoolingTimerState + .Range("AG6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
                dblCoolingOnOff = dblCoolingOnOff + .Range("AH6").Offset(intMinute + 1440 * (lngCurrentDwellingIndex - 1)).Value
            End With
        Next lngCurrentDwellingIndex
        
        ' // Store the values, normalising and converting the variables where appropriate
        aOccupancy(intMinute, 1) = dblCumulativeOccupancy / intTotalPopulation
        aActivity(intMinute, 1) = dblCumulativeActivity / intTotalPopulation
        aP_e(intMinute, 1) = dblP_e / 1000
        aP_pv(intMinute, 1) = dblP_pv / 1000
        aP_net(intMinute, 1) = (dblP_e - dblP_pv) / 1000
        aPhi_hSpace(intMinute, 1) = dblPhi_hSpace / 1000
        aPhi_hWater(intMinute, 1) = dblPhi_hWater / 1000
        aPhi_s(intMinute, 1) = dblPhi_s / 1000
        aTheta_i(intMinute, 1) = dblTheta_i / lngMaxDwellingIndex
        aV_dhw(intMinute, 1) = dblV_dhw
        aTheta_cyl(intMinute, 1) = dblTheta_cyl / lngMaxDwellingIndex
        aSpaceTimerState(intMinute, 1) = dblSpaceTimerState / lngMaxDwellingIndex
        aHotWaterTimerState(intMinute, 1) = dblHotWaterTimerState / lngMaxDwellingIndex
        aHeatingOnOff(intMinute, 1) = dblHeatingOnOff / lngMaxDwellingIndex
        aHotWaterHeatingOnOff(intMinute, 1) = dblHotWaterHeatingOnOff / lngMaxDwellingIndex
        aM_fuel(intMinute, 1) = dblM_fuel
        aPhi_collector(intMinute, 1) = dblPhi_collector / 1000
        aP_self(intMinute, 1) = dblP_self / 1000
        aMeanV_dhw(intMinute, 1) = dblV_dhw / lngMaxDwellingIndex
        aMeanV_gas(intMinute, 1) = dblM_fuel / lngMaxDwellingIndex
        aCoolingTimerState(intMinute, 1) = dblCoolingTimerState / lngMaxDwellingIndex
        aCoolingOnOff(intMinute, 1) = dblCoolingOnOff / lngMaxDwellingIndex
    Next intMinute
        
    ' // Write the aggregated results to the worksheet
    With wsResultsAggregated
        .Range("C5:C1444") = aOccupancy
        .Range("D5:D1444") = aActivity
        .Range("E5:E1444") = aP_e
        .Range("F5:F1444") = aP_pv
        .Range("G5:G1444") = aP_net
        .Range("H5:H1444") = aPhi_hSpace
        .Range("I5:I1444") = aPhi_hWater
        .Range("J5:J1444") = aPhi_s
        .Range("K5:K1444") = aTheta_i
        .Range("L5:L1444") = aV_dhw
        .Range("M5:M1444") = aTheta_cyl
        .Range("N5:N1444") = aSpaceTimerState
        .Range("O5:O1444") = aHotWaterTimerState
        .Range("P5:P1444") = aHeatingOnOff
        .Range("Q5:Q1444") = aHotWaterHeatingOnOff
        .Range("R5:R1444") = aM_fuel
        .Range("S5:S1444") = aPhi_collector
        .Range("T5:T1444") = aP_self
        .Range("U5:U1444") = aMeanV_dhw
        .Range("V5:V1444") = aMeanV_gas
        .Range("W5:W1444") = aCoolingTimerState
        .Range("X5:X1444") = aCoolingOnOff
    End With
    
End Sub




' ===============================================================================================
' ## DailyTotals
' Calculate and write daily sums for each dwelling
' ===============================================================================================
Private Sub DailyTotals(dwellingIndex As Integer, runNumber As Integer, currentDate As Date)
    Dim intRowOffset As Integer
    Dim intCurrentDwellingIndex As Integer
    Dim intRunNumber As Integer
    Dim dblMeanActiveOccupancy As Double
    Dim dblProportionActivelyOccupied As Double
    Dim dblLightingDemand As Double
    Dim dblApplianceDemand As Double
    Dim dblPVOutput As Double
    Dim dblTotalElectricityDemand As Double
    Dim dblNetElectricityDemand As Double
    Dim dblSelfConsumption As Double
    Dim dblHotWaterDemand As Double
    Dim dblAverageIndoorTemperature As Double
    Dim dblThermalEnergySpace As Double
    Dim dblThermalEnergyWater As Double
    Dim dblGasDemand As Double
        
    intRowOffset = 4
    intCurrentDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    
    ' // calculate the daily totals for the dwelling
    
    dblMeanActiveOccupancy = aOccupancy(intRunNumber).GetMeanActiveOccupancy
    dblProportionActivelyOccupied = aOccupancy(intRunNumber).GetPrActivelyOccupied
    dblLightingDemand = aLighting(intRunNumber).GetDailySumLighting / 60 / 1000
    dblApplianceDemand = aAppliances(intRunNumber).GetDailySumApplianceDemand / 60 / 1000
    
    dblTotalElectricityDemand = dblLightingDemand + dblApplianceDemand
    dblPVOutput = aPVSystem(intRunNumber).GetDailySumPvOutput / 60 / 1000
    dblNetElectricityDemand = aPVSystem(intRunNumber).GetDailySumP_net / 60 / 1000
    
    dblSelfConsumption = aPVSystem(intRunNumber).GetDailySumP_self / 60 / 1000
    
    dblHotWaterDemand = aHotWater(intRunNumber).GetDailySumHotWaterDemand
    
    dblAverageIndoorTemperature = aBuilding(intRunNumber).GetMeanTheta_i
    
    dblThermalEnergySpace = aPrimaryHeatingSystem(intRunNumber).GetDailySumThermalEnergySpace / 60 / 1000
    dblThermalEnergyWater = aPrimaryHeatingSystem(intRunNumber).GetDailySumThermalEnergyWater / 60 / 1000
    
    dblGasDemand = aPrimaryHeatingSystem(intRunNumber).GetDailySumFuelFlow / 60
        
    ' // write the variables to the worksheet
    With wsResultsDailySums
        .Cells(intRowOffset + intCurrentDwellingIndex, 1) = intCurrentDwellingIndex
        .Cells(intRowOffset + intCurrentDwellingIndex, 2) = currentDate
        .Cells(intRowOffset + intCurrentDwellingIndex, 3) = dblMeanActiveOccupancy
        .Cells(intRowOffset + intCurrentDwellingIndex, 4) = dblProportionActivelyOccupied
        .Cells(intRowOffset + intCurrentDwellingIndex, 5) = dblLightingDemand
        .Cells(intRowOffset + intCurrentDwellingIndex, 6) = dblApplianceDemand
        .Cells(intRowOffset + intCurrentDwellingIndex, 7) = dblPVOutput
        .Cells(intRowOffset + intCurrentDwellingIndex, 8) = dblTotalElectricityDemand
        .Cells(intRowOffset + intCurrentDwellingIndex, 9) = dblSelfConsumption
        .Cells(intRowOffset + intCurrentDwellingIndex, 10) = dblNetElectricityDemand
        .Cells(intRowOffset + intCurrentDwellingIndex, 11) = dblHotWaterDemand
        .Cells(intRowOffset + intCurrentDwellingIndex, 12) = dblAverageIndoorTemperature
        .Cells(intRowOffset + intCurrentDwellingIndex, 13) = dblThermalEnergySpace
        .Cells(intRowOffset + intCurrentDwellingIndex, 14) = dblThermalEnergyWater
        .Cells(intRowOffset + intCurrentDwellingIndex, 15) = dblGasDemand
        .Cells(intRowOffset + intCurrentDwellingIndex, 16) = aHeatingControls(intRunNumber).GetSpaceThermostatSetpoint
        .Cells(intRowOffset + intCurrentDwellingIndex, 17) = aSolarThermal(intRunNumber).GetDailySumPhi_s / 60 / 1000
    End With
End Sub




' ===============================================================================================
' ## AssignDwellingParameters
' Stochastically assign dwelling parameters
' ===============================================================================================
Private Sub AssignDwellingParameters(dwellingIndex As Integer)

    Dim intRowOffset As Integer
    Dim rLastCell As Range
    Dim lngLastRow As Long
    
    Dim intRow As Integer
    Dim lngRow As Long
    Dim dblRand As Double
    Dim dblCumulativeP As Double
    
    Dim intCurrentDwellingIndex As Integer
    
    Dim intMaxNumberResidents As Integer
    Dim lngMaxBuildingIndex As Long
    Dim lngMaxPrimaryHeatingSystemIndex As Long
    Dim lngMaxPvSystemIndex As Long
    Dim lngMaxSolarThermalIndex As Long
    Dim lngMaxCoolingSystemIndex As Long
    
    Dim intDwellingNumberResidents As Integer
    Dim lngDwellingBuildingIndex As Long
    Dim lngDwellingPrimaryHeatingSystemIndex As Long
    Dim lngDwellingPvSystemIndex As Long
    Dim lngDwellingSolarThermalIndex As Long
    Dim lngDwellingCoolingSystemIndex As Long
    
    intRowOffset = 4
    intCurrentDwellingIndex = dwellingIndex
    ' // determine the max possible value or index for each parameter
    intMaxNumberResidents = 5
        
    With wsBuildings
        Set rLastCell = .Cells(.Rows.Count, 1).End(xlUp)
        lngLastRow = rLastCell.Row
        lngMaxBuildingIndex = lngLastRow - intRowOffset
    End With
    
    With wsPrimaryHeatingSystems
        Set rLastCell = .Cells(.Rows.Count, 1).End(xlUp)
        lngLastRow = rLastCell.Row
        lngMaxPrimaryHeatingSystemIndex = lngLastRow - intRowOffset
    End With
    
    With wsPVSystems
        Set rLastCell = .Cells(.Rows.Count, 1).End(xlUp)
        lngLastRow = rLastCell.Row
        lngMaxPvSystemIndex = lngLastRow - intRowOffset
    End With
    
    With wsSolarThermal
        Set rLastCell = .Cells(.Rows.Count, 1).End(xlUp)
        lngLastRow = rLastCell.Row
        lngMaxSolarThermalIndex = lngLastRow - intRowOffset
    End With
    
    With wsCoolingSystems
        Set rLastCell = .Cells(.Rows.Count, 1).End(xlUp)
        lngLastRow = rLastCell.Row
        lngMaxCoolingSystemIndex = lngLastRow - intRowOffset
    End With
    
    Randomize
            
    ' // determine number of residents
    dblRand = Rnd()
    dblCumulativeP = 0
    With wsActivityStats
        For intRow = 1 To intMaxNumberResidents
            dblCumulativeP = dblCumulativeP + .Range("rPrNumberResidents").Cells(intRow, 1).Value
            If dblRand < dblCumulativeP Then
                intDwellingNumberResidents = intRow
                Exit For
            End If
        Next intRow
    End With
    
    ' // Determine the building index
    dblRand = Rnd()
    dblCumulativeP = 0
    With wsBuildings
        For lngRow = 1 To lngMaxBuildingIndex
            dblCumulativeP = dblCumulativeP + .Range("rBuildingProportion").Cells(intRowOffset + lngRow, 1).Value
            If dblRand < dblCumulativeP Then
                lngDwellingBuildingIndex = lngRow
                Exit For
            End If
        Next lngRow
    End With
    
    ' // Determine the primary heating system index
    dblRand = Rnd()
    dblCumulativeP = 0
    With wsPrimaryHeatingSystems
        For lngRow = 1 To lngMaxPrimaryHeatingSystemIndex
            dblCumulativeP = dblCumulativeP + .Range("rPrimaryHeatingSystemProportion").Cells(intRowOffset + lngRow, 1).Value
            If dblRand < dblCumulativeP Then
                lngDwellingPrimaryHeatingSystemIndex = lngRow
                Exit For
            End If
        Next lngRow
    End With
    
    ' // Determine the PvSystem index
    dblRand = Rnd()
    dblCumulativeP = 0
    With wsPVSystems
        For lngRow = 1 To lngMaxPvSystemIndex
            dblCumulativeP = dblCumulativeP + .Range("rPVProportion").Cells(intRowOffset + lngRow, 1).Value
            If dblRand < dblCumulativeP Then
                lngDwellingPvSystemIndex = lngRow
                Exit For
            End If
        Next lngRow
    End With
    
    ' // Determine the Solar Thermal index
    ' // If primary heating system is a combi boiler then there cannot be a solar thermal system
    If wsPrimaryHeatingSystems.Cells(4 + lngDwellingPrimaryHeatingSystemIndex, 4) = 2 Then
        lngDwellingSolarThermalIndex = 0
    Else
        dblRand = Rnd()
        dblCumulativeP = 0
        With wsSolarThermal
            For lngRow = 1 To lngMaxSolarThermalIndex
                dblCumulativeP = dblCumulativeP + .Range("rSolarThermalProportion").Cells(intRowOffset + lngRow, 1).Value
                If dblRand < dblCumulativeP Then
                    lngDwellingSolarThermalIndex = lngRow
                    Exit For
                End If
            Next lngRow
        End With
    End If
    
    ' // Determine the cooling system index
    dblRand = Rnd()
    dblCumulativeP = 0
    With wsCoolingSystems
        For lngRow = 1 To lngMaxCoolingSystemIndex
            dblCumulativeP = dblCumulativeP + .Range("rCoolingSystemProportion").Cells(intRowOffset + lngRow, 1).Value
            If dblRand < dblCumulativeP Then
                lngDwellingCoolingSystemIndex = lngRow
                Exit For
            End If
        Next lngRow
    End With
    
    ' // Write the dwelling parameters
    With wsDwellings
        .Cells(intRowOffset + intCurrentDwellingIndex, 1).Value = intCurrentDwellingIndex
        .Cells(intRowOffset + intCurrentDwellingIndex, 2).Value = intDwellingNumberResidents
        .Cells(intRowOffset + intCurrentDwellingIndex, 3).Value = lngDwellingBuildingIndex
        .Cells(intRowOffset + intCurrentDwellingIndex, 4).Value = lngDwellingPrimaryHeatingSystemIndex
        .Cells(intRowOffset + intCurrentDwellingIndex, 5).Value = lngDwellingPvSystemIndex
        .Cells(intRowOffset + intCurrentDwellingIndex, 6).Value = lngDwellingSolarThermalIndex
        .Cells(intRowOffset + intCurrentDwellingIndex, 7).Value = lngDwellingCoolingSystemIndex
    End With
        
End Sub




' ===============================================================================================
' ## GoToCode
' Certain shape objects on 'Model Diagrams' worksheet take the user to specific locations when
' clicked
' ===============================================================================================
Public Sub GoToCode()
    Dim strShapeName As String
    strShapeName = Application.Caller
    Select Case strShapeName
        Case "shpClimateModel1", "shpClimateModel2", "shpClimateModel3", "shpClimateModel4", "shpClimateModel5"
            Application.Goto ActiveSheet.Range("A408"), True
            
        Case "shpThermal1", "shpThermal2", "shpThermal3"
            Application.Goto ActiveSheet.Range("A46"), True
        
        Case "shpHotWater1", "shpHotWater2", "shpHotWater3"
            Application.Goto ActiveSheet.Range("A186"), True
        
        Case "shpWater1", "shpWater2"
            Application.Goto ActiveSheet.Range("A348"), True
        
        Case "shpOccupancy1"
            Application.Goto ActiveSheet.Range("A219"), True
        
        Case "shpPV1"
            Application.Goto ActiveSheet.Range("A379"), True
        
        Case "shpElectrical1"
            Application.Goto ActiveSheet.Range("A251"), True
        
        Case "shpDisaggregated1", "shpDisaggregated2", "shpDisaggregated3"
            Application.Goto ThisWorkbook.Sheets("Main Sheet").Range("A44"), True
        
        Case "shpAggregated1"
            Application.Goto ThisWorkbook.Sheets("Main Sheet").Range("A24"), True
        
        Case "shpControlSettings1", "shpControlSettings2", "shpControlSettings3", "shpControlSettings4"
            Application.Goto ThisWorkbook.Sheets("HeatingControls").Range("A1"), True
        
        Case "shpTPM1", "shpTPM2"
            Application.Goto ThisWorkbook.Sheets("tpm1_wd").Range("A1"), True
        
        Case "shpActivity1", "shpActivity2", "shpActivity3", "shpActivity4"
            Application.Goto ThisWorkbook.Sheets("ActivityStats").Range("A1"), True
        
        Case "shpBuilding1"
            Application.Goto ActiveSheet.Range("A78"), True
            
        Case "shpHeatingControls1"
            Application.Goto ActiveSheet.Range("A113"), True
            
        Case "shpHeating1"
            Application.Goto ActiveSheet.Range("A147"), True
        
        Case "shpLighting1"
            Application.Goto ActiveSheet.Range("A314"), True
        
        Case "shpAppliance1"
            Application.Goto ActiveSheet.Range("A282"), True
            
        Case "shpDwellings1", "shpDwellings2", "shpDwellings3", "shpDwellings4", "shpDwellings5", "shpDwellings6"
            Application.Goto ThisWorkbook.Sheets("Dwellings").Range("A1"), True
            
        Case Else
            Application.Goto "RunThermalElectricalDemandModel", False
            
        End Select
End Sub



' ===============================================================================================
' Code Module Functions

' ===============================================================================================
' ## GetMonteCarloNormalDistGuess
' Guess a value from a normal distribution
' ===============================================================================================
Public Function GetMonteCarloNormalDistGuess(dMean As Double, dSD As Double) As Integer
    
    Dim iGuess As Integer
    Dim bOK As Boolean
    Dim px As Double
    
    bOK = IIf(dMean = 0, True, False)
    
    iGuess = 0

    Do While (Not bOK)

        ' // Guess a value
        iGuess = (Rnd() * (dSD * 8)) - (dSD * 4) + dMean

        ' // See if this is likely
        px = (1 / (dSD * ((2 * 3.14159265359) ^ 0.5))) * Math.Exp(-((iGuess - dMean) ^ 2) / (2 * dSD * dSD))
        
        ' // End the loop if this value is okay
        If (px >= Rnd()) Then bOK = True

    Loop
    
    ' // Return the value
    GetMonteCarloNormalDistGuess = iGuess
    
End Function

