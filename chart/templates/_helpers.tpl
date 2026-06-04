{{/*
Expand the name of the chart.
*/}}
{{- define "ai-quickstart-template.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ai-quickstart-template.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ai-quickstart-template.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ai-quickstart-template.labels" -}}
helm.sh/chart: {{ include "ai-quickstart-template.chart" . }}
{{ include "ai-quickstart-template.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ai-quickstart-template.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ai-quickstart-template.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ai-quickstart-template.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ai-quickstart-template.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image name helper
*/}}
{{- define "ai-quickstart-template.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repository := .Values.global.imageRepository -}}
{{- $name := .name -}}
{{- $tag := .tag | default .Values.global.imageTag -}}
{{- printf "%s/%s/%s:%s" $registry $repository $name $tag -}}
{{- end }}

{{/*
API labels
*/}}
{{- define "ai-quickstart-template.api.labels" -}}
{{ include "ai-quickstart-template.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "ai-quickstart-template.api.selectorLabels" -}}
{{ include "ai-quickstart-template.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
UI labels
*/}}
{{- define "ai-quickstart-template.ui.labels" -}}
{{ include "ai-quickstart-template.labels" . }}
app.kubernetes.io/component: ui
{{- end }}

{{/*
UI selector labels
*/}}
{{- define "ai-quickstart-template.ui.selectorLabels" -}}
{{ include "ai-quickstart-template.selectorLabels" . }}
app.kubernetes.io/component: ui
{{- end }}

{{/*
Database labels
*/}}
{{- define "ai-quickstart-template.database.labels" -}}
{{ include "ai-quickstart-template.labels" . }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Database selector labels
*/}}
{{- define "ai-quickstart-template.database.selectorLabels" -}}
{{ include "ai-quickstart-template.selectorLabels" . }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Migration labels
*/}}
{{- define "ai-quickstart-template.migration.labels" -}}
{{ include "ai-quickstart-template.labels" . }}
app.kubernetes.io/component: migration
{{- end }}

{{/*
Migration selector labels
*/}}
{{- define "ai-quickstart-template.migration.selectorLabels" -}}
{{ include "ai-quickstart-template.selectorLabels" . }}
app.kubernetes.io/component: migration
{{- end }}

{{/*
Validate required values
*/}}
{{- define "ai-quickstart-template.validateValues" -}}
{{- required "A valid .Values.secrets.API_TOKEN is required. Set it with --set secrets.API_TOKEN=<your-maas-token>" .Values.secrets.API_TOKEN -}}
{{- end }}

{{/*
Derive API base URL from route configuration
*/}}
{{- define "ai-quickstart-template.apiBaseUrl" -}}
{{- if .Values.secrets.VITE_API_BASE_URL -}}
{{- .Values.secrets.VITE_API_BASE_URL -}}
{{- else if .Values.routes.enabled -}}
{{- $host := "" -}}
{{- if .Values.routes.sharedHost -}}
{{- $host = .Values.routes.sharedHost -}}
{{- else if .Values.routes.api.host -}}
{{- $host = .Values.routes.api.host -}}
{{- else if .Values.routes.ui.host -}}
{{- $host = .Values.routes.ui.host -}}
{{- end -}}
{{- if $host -}}
{{- if .Values.routes.api.tls.enabled -}}
{{- printf "https://%s/api" $host -}}
{{- else -}}
{{- printf "http://%s/api" $host -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}
